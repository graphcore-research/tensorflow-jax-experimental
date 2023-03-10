/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using UtilTest = HloTestBase;

TEST_F(UtilTest, TestGetUnusedParametersInCall) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_weights0, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    auto unused_or = GetUnusedParametersInCall(pipeline_stage_0);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_THAT(unused, ::testing::ElementsAre(1));
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    auto unused_or = GetUnusedParametersInCall(pipeline_stage_1);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_TRUE(unused.empty());
  }
}

TEST_F(UtilTest, TestGetUnusedCallOutputIndices) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_add, stage_0_fwd_weights0, stage_0_fwd_weights1)
}

stage_1_fwd {
  stage_1_fwd_in0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_add = f32[1,4,4,2] add(stage_1_fwd_in0, stage_1_fwd_weights2)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_add)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_add = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=2
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_stage_0_add, pipeline_weights2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd_out = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_out)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    auto unused_or = GetUnusedCallOutputIndices(pipeline_stage_0);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_THAT(unused.size(), 1);
    EXPECT_TRUE(unused.contains(1));
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    auto unused_or = GetUnusedCallOutputIndices(pipeline_stage_1);
    EXPECT_TRUE(unused_or.ok());
    auto unused = unused_or.ValueOrDie();
    EXPECT_TRUE(unused.empty());
  }
}

TEST_F(UtilTest, TestGetDuplicateCallOutputs) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_add, stage_0_fwd_weights0, stage_0_fwd_weights1, stage_0_fwd_weights0, stage_0_fwd_add, stage_0_fwd_add)
}

stage_1_fwd {
  stage_1_fwd_in0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_add = f32[1,4,4,2] add(stage_1_fwd_in0, stage_1_fwd_weights2)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_add)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_add = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=2
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_stage_0_add, pipeline_weights2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_fwd_out = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_out)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    auto duplicate_or = GetDuplicateCallOutputs(pipeline_stage_0);
    EXPECT_TRUE(duplicate_or.ok());
    auto duplicate = duplicate_or.ValueOrDie();
    EXPECT_THAT(duplicate.size(), 2);
    EXPECT_THAT(duplicate[0].size(), 2);
    EXPECT_TRUE(duplicate[0].contains(4));
    EXPECT_TRUE(duplicate[0].contains(5));
    EXPECT_THAT(duplicate[1].size(), 1);
    EXPECT_TRUE(duplicate[1].contains(3));
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    auto duplicate_or = GetDuplicateCallOutputs(pipeline_stage_1);
    EXPECT_TRUE(duplicate_or.ok());
    auto duplicate = duplicate_or.ValueOrDie();
    EXPECT_TRUE(duplicate.empty());
  }
}

TEST_F(UtilTest, TestGetDuplicateCallInputs) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_0_fwd_weights0_dupl = f32[1,4,4,2] parameter(2)
  stage_0_fwd_add = f32[1,4,4,2] add(stage_0_fwd_weights0, stage_0_fwd_weights1)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_add, stage_0_fwd_weights0_dupl)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_out = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_out, pipeline_weights2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  {
    auto duplicate_or = GetDuplicateCallInputs(pipeline_stage_0);
    EXPECT_TRUE(duplicate_or.ok());
    auto duplicate = duplicate_or.ValueOrDie();
    EXPECT_THAT(duplicate.size(), 1);
    EXPECT_THAT(duplicate[0].size(), 1);
    EXPECT_TRUE(duplicate[0].contains(2));
  }
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");
  {
    auto duplicate_or = GetDuplicateCallInputs(pipeline_stage_1);
    EXPECT_TRUE(duplicate_or.ok());
    auto duplicate = duplicate_or.ValueOrDie();
    EXPECT_TRUE(duplicate.empty());
  }
}

TEST_F(UtilTest, TestRemoveParametersFromCall) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_const = f32[] parameter(1)
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_weights0)
}

stage_1_fwd {
  stage_1_const2 = f32[] parameter(2)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_const1 = f32[] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2]) tuple(stage_1_fwd_weights1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_const1 = f32[] constant(0.01)
  pipeline_const2 = f32[] constant(0.01)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0, pipeline_const1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2]) call(pipeline_const1, pipeline_weights1, pipeline_const2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_1_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");

  HloInstruction* pipeline_const1 = FindInstruction(module0, "pipeline_const1");
  EXPECT_THAT(
      pipeline_const1->users(),
      ::testing::UnorderedElementsAre(pipeline_stage_0, pipeline_stage_1));

  HloInstruction* pipeline_const2 = FindInstruction(module0, "pipeline_const2");
  EXPECT_THAT(pipeline_const2->users(),
              ::testing::UnorderedElementsAre(pipeline_stage_1));

  HloInstruction* pipeline_weights0 =
      FindInstruction(module0, "pipeline_weights0");
  EXPECT_THAT(pipeline_stage_0->operands(),
              ::testing::ElementsAre(pipeline_weights0, pipeline_const1));
  auto new_stage_0_or = RemoveParametersFromCall(pipeline_stage_0, {1});
  EXPECT_TRUE(new_stage_0_or.ok());
  auto new_stage_0 = new_stage_0_or.ValueOrDie();
  EXPECT_THAT(new_stage_0->operands(),
              ::testing::ElementsAre(pipeline_weights0));

  EXPECT_THAT(pipeline_const1->users(),
              ::testing::UnorderedElementsAre(pipeline_stage_1));
  HloInstruction* pipeline_weights1 =
      FindInstruction(module0, "pipeline_weights1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::ElementsAre(pipeline_const1, pipeline_weights1,
                                     pipeline_const2));
  auto new_stage_1_or = RemoveParametersFromCall(pipeline_stage_1, {0, 2});
  EXPECT_TRUE(new_stage_1_or.ok());
  auto new_stage_1 = new_stage_1_or.ValueOrDie();
  EXPECT_THAT(new_stage_1->operands(),
              ::testing::ElementsAre(pipeline_weights1));
}

TEST_F(UtilTest, TestRemoveParametersFromCallInstructionsClonedCallback) {
  std::string hlo = R"(
HloModule top

comp {
  param2 = f32[4,4] parameter(0)
  param3 = f32[4,4] parameter(1)
  param4 = f32[4,4] parameter(2)
  ROOT dot = f32[4,4] dot(param2, param3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  param0 = f32[4,4] parameter(0), parameter_replication={false}
  param1 = f32[4,4] parameter(1), parameter_replication={false}
  ROOT call = f32[4,4] call(param0, param1, param1), to_apply=comp
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  bool callback_called = false;
  std::function<void(const HloCloneContext*)> callback =
      [&](const HloCloneContext* context) -> void {
    // Iterate over all cloned calls in the HloCloneContext.
    auto& cloned_instructions = context->cloned_instructions();
    // Cloned call + 3 out of the 4 instructions in the original computation.
    EXPECT_EQ(cloned_instructions.size(), 4);

    // Find the entry for the call instruction.
    auto it = absl::c_find_if(
        cloned_instructions,
        [](const std::pair<const HloInstruction*, const HloInstruction*>&
               pair) { return pair.first->opcode() == HloOpcode::kCall; });
    EXPECT_NE(it, cloned_instructions.end());
    auto pair = *it;
    EXPECT_EQ(pair.second->opcode(), HloOpcode::kCall);

    // Check that all instructions still exist (haven't been deleted yet).
    const HloInstruction* old_call = pair.first;
    const HloInstruction* new_call = pair.second;
    EXPECT_EQ(old_call->operand_count(), 3);
    EXPECT_EQ(new_call->operand_count(), 2);
    EXPECT_EQ(old_call->to_apply()->instruction_count(), 4);
    EXPECT_EQ(new_call->to_apply()->instruction_count(), 3);
    EXPECT_TRUE(Match(old_call->to_apply()->root_instruction(),
                      m::Dot(m::Parameter(0), m::Parameter(1))));
    EXPECT_TRUE(
        Match(old_call->to_apply()->parameter_instruction(2), m::Parameter(2)));
    EXPECT_TRUE(Match(new_call->to_apply()->root_instruction(),
                      m::Dot(m::Parameter(0), m::Parameter(1))));
    callback_called = true;
  };

  HloCloneContext context(module0);
  HloInstruction* call = FindInstruction(module0, "call");
  auto new_call_or = RemoveParametersFromCall(call, {2}, &context, callback);
  EXPECT_TRUE(new_call_or.ok());
  EXPECT_TRUE(callback_called);
}

TEST_F(UtilTest, TestAddParametersToCallInstructionsClonedCallback) {
  std::string hlo = R"(
HloModule top

comp {
  param2 = f32[4,4] parameter(0)
  param3 = f32[4,4] parameter(1)
  ROOT dot = f32[4,4] dot(param2, param3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  param0 = f32[4,4] parameter(0), parameter_replication={false}
  param1 = f32[4,4] parameter(1), parameter_replication={false}
  ROOT call = f32[4,4] call(param0, param1), to_apply=comp
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  bool callback_called = false;
  std::function<void(const HloCloneContext*)> callback =
      [&](const HloCloneContext* context) -> void {
    // Iterate over all cloned calls in the HloCloneContext.
    auto& cloned_instructions = context->cloned_instructions();
    // Cloned call + the 3 instructions in the original computation.
    EXPECT_EQ(cloned_instructions.size(), 4);

    // Find the entry for the call instruction.
    auto it = absl::c_find_if(
        cloned_instructions,
        [](const std::pair<const HloInstruction*, const HloInstruction*>&
               pair) { return pair.first->opcode() == HloOpcode::kCall; });
    EXPECT_NE(it, cloned_instructions.end());
    auto pair = *it;
    EXPECT_EQ(pair.second->opcode(), HloOpcode::kCall);

    // Check that all instructions still exist (haven't been deleted yet).
    const HloInstruction* old_call = pair.first;
    const HloInstruction* new_call = pair.second;
    EXPECT_EQ(old_call->operand_count(), 2);
    EXPECT_EQ(new_call->operand_count(), 3);
    EXPECT_EQ(old_call->to_apply()->instruction_count(), 3);
    EXPECT_EQ(new_call->to_apply()->instruction_count(), 4);
    EXPECT_TRUE(Match(old_call->to_apply()->root_instruction(),
                      m::Dot(m::Parameter(0), m::Parameter(1))));
    EXPECT_TRUE(Match(new_call->to_apply()->root_instruction(),
                      m::Dot(m::Parameter(0), m::Parameter(1))));
    EXPECT_TRUE(
        Match(new_call->to_apply()->parameter_instruction(2), m::Parameter(2)));
    callback_called = true;
  };

  HloCloneContext context(module0);
  HloInstruction* call = FindInstruction(module0, "call");
  HloInstruction* param1 = FindInstruction(module0, "param1");
  auto new_call_or = AddParametersToCall(call, {param1}, &context, callback);
  EXPECT_TRUE(new_call_or.ok());
  EXPECT_TRUE(callback_called);
}

TEST_F(UtilTest, TestRemoveOutputsFromCall) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_const = f32[] parameter(1)
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[]) tuple(stage_0_fwd_weights0, stage_0_const)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_const = f32[] parameter(2)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights1, stage_1_const)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_const = f32[] constant(0.01)
  pipeline_stage_0 = (f32[1,4,4,2], f32[]) call(pipeline_weights0, pipeline_const), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(pipeline_stage_0_w0, pipeline_weights1, pipeline_const), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_0_w0, pipeline_stage_1_w1)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  auto module = ParseAndReturnVerifiedModule(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  HloInstruction* pipeline_stage_0 =
      FindInstruction(module0, "pipeline_stage_0");
  HloInstruction* pipeline_stage_1 =
      FindInstruction(module0, "pipeline_stage_1");

  HloInstruction* pipeline_const = FindInstruction(module0, "pipeline_const");
  EXPECT_THAT(pipeline_const->users(), ::testing::UnorderedElementsAre(
                                           pipeline_stage_0, pipeline_stage_1));

  HloInstruction* pipeline_weights0 =
      FindInstruction(module0, "pipeline_weights0");
  EXPECT_THAT(pipeline_stage_0->operands(),
              ::testing::ElementsAre(pipeline_weights0, pipeline_const));
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_0->shape()), 2);
  EXPECT_TRUE(RemoveOutputsFromCall(pipeline_stage_0, {1}).ok());
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_0->shape()), 1);
  EXPECT_THAT(pipeline_stage_0->user_count(), 1);

  HloInstruction* pipeline_stage_0_w0 = pipeline_stage_0->users()[0];
  HloInstruction* pipeline_weights1 =
      FindInstruction(module0, "pipeline_weights1");
  EXPECT_THAT(pipeline_stage_1->operands(),
              ::testing::ElementsAre(pipeline_stage_0_w0, pipeline_weights1,
                                     pipeline_const));
  EXPECT_THAT(pipeline_stage_1->user_count(), 1);
  HloInstruction* pipeline_stage_1_w1 = pipeline_stage_1->users()[0];
  EXPECT_THAT(pipeline_stage_1_w1->tuple_index(), 1);
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_1->shape()), 3);
  EXPECT_TRUE(RemoveOutputsFromCall(pipeline_stage_1, {0, 2}).ok());
  EXPECT_THAT(ShapeUtil::TupleElementCount(pipeline_stage_1->shape()), 1);
  EXPECT_THAT(pipeline_stage_1_w1->tuple_index(), 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
