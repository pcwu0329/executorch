# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from examples.arm.QAT_example.rife_vgf import (
    configure_rife_vgf,
    configure_rife_vgf_quantizer,
)
from executorch.backends.arm.quantizer import get_vgf_snorm_quantization_config
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.quantize import ArmQuantize
from executorch.backends.arm.test.tester.test_pipeline import VgfPipeline
from executorch.backends.arm.vgf import VgfPartitioner


# Form a grid and call the torch.nn.functional.grid_sample
# The grid_sample is implemented as custom shader(TOSA CUSTOM) in the graph
def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    horizontal = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    vertical = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    base_grid = torch.cat(
        (
            horizontal.expand(1, -1, height, -1),
            vertical.expand(1, -1, -1, width),
        ),
        dim=1,
    )
    grid = (base_grid + flow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def warp_downsample(
    image: torch.Tensor, flow: torch.Tensor, scale: int
) -> torch.Tensor:
    # Note that we call the torch.ops.rife.warp_downsample
    # rather than the standard torch.nn.functional.interpolate. In this way,
    # we run the downsample also aqs a shader operation on the EE.
    if scale == 2:
        return torch.ops.rife.warp_downsample2.default(image, flow)
    if scale == 4:
        return torch.ops.rife.warp_downsample4.default(image, flow)
    if scale == 8:
        return torch.ops.rife.warp_downsample8.default(image, flow)
    raise ValueError(f"Unsupported warp downsample scale: {scale}")


def warp_downsample8_cpu(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        warp(image, flow), scale_factor=1.0 / 8.0, mode="bilinear", align_corners=False
    )


class TinyRifeConv(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )


class TinyRifeBlock(torch.nn.Module):
    def __init__(self, in_channels: int, scale: int, output_feat: bool = True) -> None:
        super().__init__()
        channels = 8
        self.scale = scale
        self.conv0 = torch.nn.Sequential(
            TinyRifeConv(in_channels, channels // 2, stride=2),
            TinyRifeConv(channels // 2, channels, stride=2),
        )
        self.convblock = torch.nn.Sequential()
        for _ in range(8):
            self.convblock.append(TinyRifeConv(channels, channels))
        output_channels = 13 if output_feat else 5
        self.lastconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channels, 4 * output_channels, 4, 2, 1),
            torch.nn.PixelShuffle(2),
        )

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 16:
            x = torch.nn.functional.interpolate(
                x,
                scale_factor=1.0 / 8.0,
                mode="bilinear",
                align_corners=False,
            )
            return torch.nn.functional.interpolate(
                x,
                scale_factor=1.0 / 2.0,
                mode="bilinear",
                align_corners=False,
            )
        return torch.nn.functional.interpolate(
            x,
            scale_factor=1.0 / self.scale,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        native_scale_inputs: tuple[torch.Tensor, ...] = (),
        trailing_inputs: tuple[torch.Tensor, ...] = (),
        output_scale: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = [self.downsample(tensor) for tensor in inputs]
        tensors.extend(native_scale_inputs)
        tensors.extend(self.downsample(tensor) for tensor in trailing_inputs)
        x = torch.cat(tensors, dim=1)
        timestep = x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), 0.5)
        x = torch.cat((x[:, :14], timestep, x[:, 14:]), dim=1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        flow = torch.nn.functional.interpolate(
            tmp[:, :4], scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        output_resize = float(self.scale) / output_scale
        mask = torch.nn.functional.interpolate(
            tmp[:, 4:5],
            scale_factor=output_resize,
            mode="bilinear",
            align_corners=False,
        )
        feat = torch.nn.functional.interpolate(
            tmp[:, 5:],
            scale_factor=output_resize,
            mode="bilinear",
            align_corners=False,
        )
        return flow, mask, feat


class TinyRifeHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(4, 4, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyRife(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode = TinyRifeHead()
        self.block0 = TinyRifeBlock(15, scale=16)
        self.block1 = TinyRifeBlock(28, scale=8)
        self.block2 = TinyRifeBlock(28, scale=1, output_feat=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image0, image1 = x[:, :3], x[:, 3:]
        feat0, feat1 = self.encode(image0), self.encode(image1)
        flow, mask, feat = self.block0(
            (image0, image1, feat0, feat1), output_scale=self.block1.scale
        )

        warped0 = warp_downsample(image0, flow[:, :2], 8)
        warped1 = warp_downsample(image1, flow[:, 2:4], 8)
        warped_feat0 = warp_downsample(feat0, flow[:, :2], 8)
        warped_feat1 = warp_downsample(feat1, flow[:, 2:4], 8)
        delta, mask, feat = self.block1(
            (),
            native_scale_inputs=(
                warped0,
                warped1,
                warped_feat0,
                warped_feat1,
                mask,
                feat,
            ),
            trailing_inputs=(flow,),
            output_scale=self.block2.scale,
        )
        flow = flow + delta

        warped0 = warp(image0, flow[:, :2])
        warped1 = warp(image1, flow[:, 2:4])
        warped_feat0 = warp(feat0, flow[:, :2])
        warped_feat1 = warp(feat1, flow[:, 2:4])
        delta, mask, _ = self.block2(
            (warped0, warped1, warped_feat0, warped_feat1),
            native_scale_inputs=(mask, feat),
            trailing_inputs=(flow,),
        )
        flow = flow + delta
        warped0 = warp(image0, flow[:, :2])
        warped1 = warp(image1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)


@common.SkipIfNoModelConverter
def test_tiny_rife_folded_qdq_vgf() -> None:
    test_data: tuple[torch.Tensor] = (
        torch.rand(1, 6, 64, 64).contiguous(memory_format=torch.channels_last),
    )
    pipeline = VgfPipeline[tuple[torch.Tensor]](
        TinyRife().eval(),
        test_data,
        aten_op=[],
        run_on_vulkan_runtime=False,
        symmetric_io_quantization=True,
        preserve_io_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    quantization_config = get_vgf_snorm_quantization_config()
    pipeline.quantizer.set_global(quantization_config)
    pipeline.quantizer.set_io(quantization_config)
    configure_rife_vgf_quantizer(pipeline.quantizer)
    pipeline.change_args(
        "quantize", ArmQuantize(pipeline.quantizer, quantization_config)
    )
    pipeline.pop_stage("check_not.exir_quant_nodes")
    partitioner = VgfPartitioner(pipeline.quantizer.compile_spec)
    configure_rife_vgf(partitioner)
    warp_downsample_library = torch.library.Library("rife", "IMPL", "CPU")
    warp_downsample_library.impl("warp_downsample8", warp_downsample8_cpu)
    pipeline.change_args("to_edge_transform_and_lower", partitioners=[partitioner])
    pipeline.run()
