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

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/feed_token_verifier.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {
bool HasUniqueToken(const HloInstruction* inst) {
  EXPECT_TRUE(inst);
  if (inst->opcode() == HloOpcode::kInfeed) {
    const auto* token = inst->operand(0);
    EXPECT_TRUE(token->shape().IsToken());

    if (token->opcode() == HloOpcode::kAfterAll && token->user_count() == 1) {
      return true;
    }
  } else if (inst->opcode() == HloOpcode::kOutfeed) {
    const auto* token = inst->operand(1);
    EXPECT_TRUE(token->shape().IsToken());

    if (token->opcode() == HloOpcode::kAfterAll && token->user_count() == 1) {
      return true;
    }
  }
  return false;
}

using FeedTokenVerifierTest = HloTestBase;

TEST_F(FeedTokenVerifierTest, BasicTest) {
  std::string hlo = R"(
HloModule m

ENTRY entry {
  after-all = token[] after-all()
  infeed = ((f32[1]{0}), token[]) infeed(token[] after-all)

  gte0 = (f32[1]{0}) get-tuple-element(((f32[1]{0}), token[]) infeed), index=0
  gte1 = token[] get-tuple-element(((f32[1]{0}), token[]) infeed), index=1

  outfeed0 = token[] outfeed((f32[1]{0}) gte0, token[] gte1)
  outfeed1 = token[] outfeed((f32[1]{0}) gte0, token[] after-all)

  ROOT root = () tuple()
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  EXPECT_TRUE(FeedTokenVerifier().Run(module.get()).ValueOrDie());

  auto* infeed = FindInstruction(module.get(), "infeed");
  auto* outfeed0 = FindInstruction(module.get(), "outfeed0");
  auto* outfeed1 = FindInstruction(module.get(), "outfeed1");

  EXPECT_TRUE(HasUniqueToken(infeed));
  EXPECT_TRUE(HasUniqueToken(outfeed0));
  EXPECT_TRUE(HasUniqueToken(outfeed1));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
