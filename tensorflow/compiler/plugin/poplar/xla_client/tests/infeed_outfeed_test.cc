/* Copyright (c) 2022 Graphcore Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_compiler.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

const auto platform = PlatformUtil::GetPlatform("IPU").ValueOrDie();
const auto ipu_client = GetLegacyIpuClient(true, IpuConfig()).ValueOrDie();

class InfeedOutfeedTest : public LocalClientTestBase {
 public:
  InfeedOutfeedTest() : LocalClientTestBase(platform) {}

 protected:
  void TestInfeedOutfeedRoundTrip(const Literal& literal) {
    XlaBuilder builder(TestName());

    const auto in_shape = literal.shape().IsTuple()
                              ? literal.shape()
                              : ShapeUtil::MakeTupleShape({literal.shape()});
    const auto in = xla::Infeed(&builder, in_shape);

    std::vector<XlaOp> gtes;
    for (auto i = 0; i < in_shape.tuple_shapes_size(); i++) {
      gtes.push_back(xla::GetTupleElement(in, i));
    }

    const auto tuple = xla::Tuple(&builder, gtes);
    xla::Outfeed(tuple, in_shape, /*outfeed_config=*/"");

    std::unique_ptr<tensorflow::Thread> thread(
        tensorflow::Env::Default()->StartThread(
            tensorflow::ThreadOptions(), "execute_thread",
            [&] { ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {}); }));

    ASSERT_IS_OK(local_client_->TransferToInfeedLocal(
        literal, local_client_->default_device_ordinal()));

    Literal result(literal.shape());
    ASSERT_IS_OK(local_client_->TransferFromOutfeedLocal(
        local_client_->default_device_ordinal(), &result));

    EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
  }
};

TEST_F(InfeedOutfeedTest, R0Bool) {
  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR0<bool>(true));
}

TEST_F(InfeedOutfeedTest, R1U32) {
  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR1<uint32>({1, 2, 3}));
}

TEST_F(InfeedOutfeedTest, R2F32) {
  TestInfeedOutfeedRoundTrip(
      LiteralUtil::CreateR2F32Linspace(0.0, 1.0, 128, 64));
}

TEST_F(InfeedOutfeedTest, R3F32) {
  TestInfeedOutfeedRoundTrip(
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

TEST_F(InfeedOutfeedTest, R3F32DifferentLayout) {
  const Layout r3_dim0minor = LayoutUtil::MakeLayout({0, 1, 2});
  const Layout r3_dim0major = LayoutUtil::MakeLayout({2, 1, 0});

  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0minor));

  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0major));
}

TEST_F(InfeedOutfeedTest, R4S32) {
  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

TEST_F(InfeedOutfeedTest, Tuple) {
  TestInfeedOutfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<uint32>({1, 2, 3}),
       LiteralUtil::CreateR0<bool>(false)}));
}

TEST_F(InfeedOutfeedTest, EmptyTuple) {
  TestInfeedOutfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

TEST_F(InfeedOutfeedTest, LargeInfeedOutfeed) {
  Array4D<float> array(80, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedOutfeedRoundTrip(LiteralUtil::CreateR4FromArray4D<float>(array));
}

TEST_F(InfeedOutfeedTest, LargeTupleInfeedOutfeed) {
  Array4D<float> array(40, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedOutfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR4FromArray4D<float>(array),
       LiteralUtil::CreateR0<int32>(5)}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
