/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_manifest_fixture.h"
#include <limits>

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

TEST(CoreAIManifestTest, AcceptsInlineAndRejectsInvalidJson) {
  EXPECT_TRUE(parse_manifest(encode(manifest_dict())).ok());
  EXPECT_FALSE(parse_manifest([@"[]" dataUsingEncoding:NSUTF8StringEncoding]).ok());
  EXPECT_FALSE(parse_manifest([@"garbage" dataUsingEncoding:NSUTF8StringEncoding]).ok());
}

TEST(CoreAIManifestTest, RejectsIncompatibleVersions) {
  for (id version in @[ @0, @1, @3, @2.5, @YES, @"2", NSNull.null ]) {
    SCOPED_TRACE([version description].UTF8String);
    auto dict = manifest_dict();
    dict[@"version"] = version;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::DelegateInvalidCompatibility);
  }
  auto dict = aot_manifest_dict();
  [dict removeObjectForKey:@"version"];
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::DelegateInvalidCompatibility);
}

TEST(CoreAIManifestTest, RejectsUnsupportedDeliveryAndState) {
  for (NSString* packaging in @[ @"unknown", @"sidecar", @"aot_compiled_sidecar" ]) {
    SCOPED_TRACE(packaging.UTF8String);
    auto dict = [packaging hasPrefix:@"aot"] ? aot_manifest_dict() : manifest_dict();
    dict[@"packaging"] = packaging;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::NotSupported);
  }
  auto dict = manifest_dict();
  dict[@"runtime_supported"] = @NO;
  EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::NotSupported);
}

TEST(CoreAIManifestTest, RejectsUnsafeSourcePaths) {
  for (NSString* path in @[
         @"../model.aimodel", @"/ab/model.aimodel", @"ab/../model.aimodel",
         @"xx/model.aimodel"
       ]) {
    SCOPED_TRACE(path.UTF8String);
    auto dict = manifest_dict();
    dict[@"path"] = path;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, RejectsInvalidFileMaps) {
  for (id files in @[
         @[], @{}, NSNull.null, @"files",
         @{@"model.aimodel/../escape" : @1},
         @{@"model.aimodel/bad\\name" : @1},
         @{@"/tmp/file" : @1},
         @{@"model.aimodel/one" : @1, @"model.aimodel/one/two" : @2}
       ]) {
    SCOPED_TRACE([files description].UTF8String);
    auto dict = manifest_dict();
    dict[@"files"] = files;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, RejectsInvalidInputNames) {
  for (id names in @[ @[ @"same", @"same" ], @[ @1 ], @[ @"" ], @"input", NSNull.null ]) {
    SCOPED_TRACE([names description].UTF8String);
    auto dict = manifest_dict();
    dict[@"input_names"] = names;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
}

TEST(CoreAIManifestTest, ValidatesDeploymentFloor) {
  for (NSString* version in @[ @"", @"27.x", @"-1", @"27.0.0.0", @"9999999999" ]) {
    SCOPED_TRACE(version.UTF8String);
    auto dict = manifest_dict();
    dict[@"min_deployment_version"] = version;
    EXPECT_FALSE(parse_manifest(encode(dict)).ok());
  }
  auto dict = manifest_dict();
  dict[@"min_deployment_version"] = NSNull.null;
  EXPECT_TRUE(parse_manifest(encode(dict)).ok());
}

class CoreAIManifestMetadataTest : public ::testing::TestWithParam<bool> {
 protected:
  NSMutableDictionary* manifest() {
    return GetParam() ? aot_manifest_dict() : manifest_dict();
  }
  NSString* file() {
    return GetParam() ? @"model.arch_b.aimodelc/graph.bin" : @"model.aimodel/graph.bin";
  }
};

TEST_P(CoreAIManifestMetadataTest, RequiresMetadataMaps) {
  auto dict = manifest();
  ASSERT_TRUE(parse_manifest(encode(dict)).ok());
  for (NSString* key in @[ @"files", @"bundle_digests" ]) {
    SCOPED_TRACE(key.UTF8String);
    NSMutableDictionary* invalid = [dict mutableCopy];
    [invalid removeObjectForKey:key];
    EXPECT_EQ(parse_manifest(encode(invalid)).error(), Error::InvalidProgram);
    for (id value in @[ NSNull.null, @[], @{}, @"invalid", @1 ]) {
      SCOPED_TRACE([value description].UTF8String);
      invalid[key] = value;
      EXPECT_EQ(parse_manifest(encode(invalid)).error(), Error::InvalidProgram);
    }
  }
}

TEST_P(CoreAIManifestMetadataTest, RejectsInvalidFileSizes) {
  for (id size in @[
         @YES, @NO, @(-1), @0.5, @11.25,
         @(std::numeric_limits<uint64_t>::max()),
         [NSDecimalNumber decimalNumberWithString:@"9223372036854775808"],
         @1e30, @"11", NSNull.null, @[], @{}
       ]) {
    SCOPED_TRACE([size description].UTF8String);
    auto dict = manifest();
    NSMutableDictionary* files = [dict[@"files"] mutableCopy];
    files[file()] = size;
    dict[@"files"] = files;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST_P(CoreAIManifestMetadataTest, AcceptsIntegralFileSizes) {
  for (NSNumber* size in @[ @0, @11, @(std::numeric_limits<int64_t>::max()) ]) {
    SCOPED_TRACE(size.description.UTF8String);
    auto dict = manifest();
    NSMutableDictionary* files = [dict[@"files"] mutableCopy];
    files[file()] = size;
    dict[@"files"] = files;
    EXPECT_TRUE(parse_manifest(encode(dict)).ok());
  }
}

TEST_P(CoreAIManifestMetadataTest, RequiresExactLowercaseDigests) {
  for (id digest in @[
         @"", @"abc", [fixture_identity('a') substringFromIndex:1],
         [fixture_identity('a') stringByAppendingString:@"a"],
         fixture_identity('a').uppercaseString,
         [@"g" stringByPaddingToLength:64 withString:@"g" startingAtIndex:0],
         @1, NSNull.null, @[], @{}
       ]) {
    SCOPED_TRACE([digest description].UTF8String);
    auto dict = manifest();
    NSMutableDictionary* digests = [dict[@"bundle_digests"] mutableCopy];
    digests[file().stringByDeletingLastPathComponent] = digest;
    dict[@"bundle_digests"] = digests;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

TEST_P(CoreAIManifestMetadataTest, RequiresExactBundleCoverage) {
  for (bool extra : {false, true}) {
    SCOPED_TRACE(extra ? "extra digest" : "missing digest");
    auto dict = manifest();
    NSMutableDictionary* digests = [dict[@"bundle_digests"] mutableCopy];
    if (extra) {
      digests[@"unexpected.aimodel"] = fixture_identity('e');
    } else {
      [digests removeObjectForKey:file().stringByDeletingLastPathComponent];
    }
    dict[@"bundle_digests"] = digests;
    EXPECT_EQ(parse_manifest(encode(dict)).error(), Error::InvalidProgram);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Delivery, CoreAIManifestMetadataTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Aot" : "Source";
    });

TEST(CoreAIManifestTest, RejectsFractionalJsonSizeSyntax) {
  NSString* json = [[NSString alloc] initWithData:encode(manifest_dict())
                                        encoding:NSUTF8StringEncoding];
  ASSERT_TRUE([json containsString:@":11"]);
  for (NSString* size in @[ @":11.0", @":11e0" ]) {
    SCOPED_TRACE(size.UTF8String);
    NSString* fractional = [json stringByReplacingOccurrencesOfString:@":11" withString:size];
    EXPECT_EQ(parse_manifest([fractional dataUsingEncoding:NSUTF8StringEncoding]).error(),
              Error::InvalidProgram);
  }
}

TEST(CoreAIManifestTest, SelectsAotArchitectureWithoutMutatingManifest) {
  auto dict = aot_manifest_dict();
  auto parsed = parse_manifest(encode(dict));
  ASSERT_TRUE(parsed.ok());
  ASSERT_TRUE(parsed->aot_compiled);
  EXPECT_EQ(parsed->path, nil);
  for (NSString* arch in @[ @"arch_a", @"arch_b" ]) {
    SCOPED_TRACE(arch.UTF8String);
    auto selected = select_assets(parsed.get(), arch, @"macOS");
    ASSERT_TRUE(selected.ok());
    EXPECT_TRUE([selected->path isEqualToString:dict[@"archs"][arch]]);
    EXPECT_EQ(selected->files.count, 2);
    NSString* bundle = selected->path.lastPathComponent;
    EXPECT_TRUE(([selected->bundle_digests isEqual:@{bundle : dict[@"bundle_digests"][bundle]}]));
    EXPECT_TRUE(([selected->files isEqual:@{
      [bundle stringByAppendingPathComponent:@"graph.bin"] : @14,
      [bundle stringByAppendingPathComponent:@"nested/weights.bin"] : @7
    }]));
    EXPECT_EQ(parsed->path, nil);
  }
}

TEST(CoreAIManifestTest, RejectsMismatchedAotPlatformAndArchitecture) {
  auto ios = aot_manifest_dict();
  ios[@"platform"] = @"iOS";
  auto parsed_ios = parse_manifest(encode(ios));
  ASSERT_TRUE(parsed_ios.ok());
  EXPECT_TRUE(select_assets(parsed_ios.get(), @"arch_b", @"iOS").ok());
  auto parsed = parse_manifest(encode(aot_manifest_dict()));
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(select_assets(parsed.get(), @"arch_b", @"iOS").error(),
            Error::DelegateInvalidCompatibility);
  for (id arch in @[ @"missing", @"ARCH_B", @"", NSNull.null ]) {
    SCOPED_TRACE([arch description].UTF8String);
    EXPECT_EQ(select_assets(parsed.get(), arch == NSNull.null ? nil : arch, @"macOS").error(),
              Error::DelegateInvalidCompatibility);
  }
}
