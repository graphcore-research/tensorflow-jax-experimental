/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Loop.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor_utils.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace xla {
namespace poplarplugin {

namespace util = ::xla::poplarplugin::pipelinevisitorutils;

namespace {
/**
 * Get the number of stages in a pipeline.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The number of stages inside the pipeline computation.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
int64_t GetPipelineStageCount(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();

  return absl::c_count_if(pipeline_computation->instructions(),
                          [](const HloInstruction* inst) {
                            return IsPipelineStageOrBackwardOp(inst);
                          });
}

/**
 * Get the pipeline stage to device mapping.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping of the ith stage to a IPU device.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
std::vector<int> GetPipelineStageDeviceMapping(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  std::vector<HloInstruction*> instructions(
      pipeline_computation->instructions().begin(),
      pipeline_computation->instructions().end());

  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stage = GetPipelineStages(pipeline_computation).ValueOrDie();
  stage.forward.insert(stage.forward.end(), stage.backward.rbegin(),
                       stage.backward.rend());

  std::vector<int> result(stage.forward.size());

  const auto get_stage_shard = [](const HloInstruction* hlo) -> int {
    return *hlo->sharding_unique_device();
  };
  absl::c_transform(stage.forward, result.begin(), get_stage_shard);

  return result;
}

/**
 * Get the pipeline instruction to stage mapping. When an instruction isn't a
 * stage call, it must be associated with a stage.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping from Hlo instructions to pipeline stage index.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
absl::flat_hash_map<const HloInstruction*, int> GetPipelineInstStageMapping(
    const HloInstruction* pipeline) {
  absl::flat_hash_map<const HloInstruction*, int> result;
  HloComputation* pipeline_computation = pipeline->to_apply();
  auto instructions = pipeline_computation->MakeInstructionPostOrder();

  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stage = GetPipelineStages(pipeline_computation).ValueOrDie();
  stage.forward.insert(stage.forward.end(), stage.backward.rbegin(),
                       stage.backward.rend());

  // Loop through all of the pipeline stage calls.
  // These trivially belong to the stage id that corresponds to their position.
  for (auto itr = stage.forward.begin(); itr != stage.forward.end(); ++itr) {
    result.insert(
        std::make_pair(*itr, std::distance(stage.forward.begin(), itr)));
  }

  // Assign the recomputation stages to the same stage as the forward stage.
  for (auto pair : stage.recomputation) {
    result[pair.second] = pair.first;
  }

  // Partition out the stage calls instructions and skip them.
  auto stages_end =
      std::stable_partition(instructions.begin(), instructions.end(),
                            util::HasHloOpcode(HloOpcode::kCall));

  // Comparison of HloInstructions with assigned stage index.
  const auto inst_comparison = [&](HloInstruction* a,
                                   HloInstruction* b) -> bool {
    return result.at(a) < result.at(b);
  };

  // Assign the root instruction to the last stage. Note that we expect the root
  // instruction to be a tuple which does not modify the sequences.
  HloInstruction* root = pipeline_computation->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  result[root] = stage.forward.size() - 1;

  // Assign the resource update stage (if there is one) to the last stage. Note
  // that it won't actually be executed with the last stage.
  if (stage.resource_update) {
    result[*stage.resource_update] = stage.forward.size() - 1;
  }

  // Get the stage given the users. Requires all the users to already have a
  // stage.
  auto get_stage_from_users = [&](const HloInstruction* inst) {
    CHECK_NE(inst->user_count(), 0);
    auto users = inst->users();
    return result.at(*absl::c_min_element(users, inst_comparison));
  };

  // Get the stage given the operands. Requires all the operands to already have
  // a stage.
  auto get_stage_from_operands = [&](const HloInstruction* inst) {
    CHECK_NE(inst->operand_count(), 0);
    auto operands = inst->operands();
    return result.at(*absl::c_max_element(operands, inst_comparison));
  };

  auto inter_tileset_copy_in_end = std::stable_partition(
      stages_end, instructions.end(), util::IsInterTilesetCopyInInstruction);
  for (auto itr = stages_end; itr != inter_tileset_copy_in_end; ++itr) {
    HloInstruction* inst = *itr;

    CHECK_EQ(inst->user_count(), 1);
    const HloInstruction* tuple = inst->users()[0];
    CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
    // Expect at least one user of GTE to be a forward stage.
    auto fwd_stage_itr = absl::c_find_if(
        tuple->users(),
        [](const HloInstruction* inst) { return IsPipelineStage(inst); });
    CHECK(fwd_stage_itr != tuple->users().end());
    int64_t stage = result.at(*fwd_stage_itr);
    result[inst] = stage;
    result[tuple] = stage;
  }

  // Partition out infeeds.
  auto infeeds_end =
      std::stable_partition(inter_tileset_copy_in_end, instructions.end(),
                            util::HasHloOpcode(HloOpcode::kInfeed));
  for (auto itr = inter_tileset_copy_in_end; itr != infeeds_end; ++itr) {
    HloInstruction* inst = *itr;
    // For an infeed, assign the stages for the infeed, its gte user, and
    // the input token.
    const HloInstruction* token = inst->operand(0);
    CHECK_EQ(inst->user_count(), 1);
    CHECK_EQ(token->opcode(), HloOpcode::kAfterAll);
    const HloInstruction* gte = inst->users()[0];
    CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);

    // Find a GTE with a non-GTE user.
    while (gte->user_count() > 0 &&
           gte->users()[0]->opcode() == HloOpcode::kGetTupleElement) {
      gte = gte->users()[0];
    }

    // Expect at least one user of GTE to be a forward stage or inter-tilset
    // copy.
    auto fwd_stage_itr =
        absl::c_find_if(gte->users(), [](const HloInstruction* inst) {
          return IsPipelineStage(inst) ||
                 util::IsInterTilesetCopyInInstruction(inst);
        });
    CHECK(fwd_stage_itr != gte->users().end());
    int64_t stage = result.at(*fwd_stage_itr);
    result[inst] = stage;
    result[gte] = stage;
    result[token] = stage;
  }

  auto inter_tileset_copy_out_end = std::stable_partition(
      infeeds_end, instructions.end(), util::IsInterTilesetCopyOutInstruction);
  for (auto itr = infeeds_end; itr != inter_tileset_copy_out_end; ++itr) {
    HloInstruction* inst = *itr;

    const HloInstruction* gte = inst->operand(0);
    CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    // Find a GTE with a non-GTE user.
    while (gte->operand(0)->opcode() == HloOpcode::kGetTupleElement) {
      gte = gte->operand(0);
    }

    const HloInstruction* call = gte->operand(0);
    CHECK_EQ(call->opcode(), HloOpcode::kCall);

    int64_t stage = result.at(call);
    result[inst] = stage;
    result[gte] = stage;
  }

  // Partition out the outfeeds.
  auto outfeeds_end = std::stable_partition(
      infeeds_end, instructions.end(), util::HasHloOpcode(HloOpcode::kOutfeed));
  for (auto itr = infeeds_end; itr != outfeeds_end; ++itr) {
    HloInstruction* inst = *itr;
    // For an outfeed, assign the stages for the outfeed, its gte operand, and
    // the input token.
    const HloInstruction* copy = inst->operand(0);
    CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
    const HloInstruction* token = inst->operand(1);
    CHECK_EQ(token->opcode(), HloOpcode::kAfterAll);
    const HloInstruction* gte = copy->operand(0);
    if (result.contains(gte)) {
      int64_t stage = result.at(gte);
      result[inst] = stage;
      result[copy] = stage;
      result[token] = stage;
    } else if (gte->opcode() == HloOpcode::kTuple) {
      const HloInstruction* gte2 = gte->operand(0);
      CHECK(util::IsInterTilesetCopyOutInstruction(gte2));
      int64_t stage = result.at(gte2);
      result[inst] = stage;
      result[gte] = stage;
      result[gte2] = stage;
      result[token] = stage;
    } else {
      CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      int64_t stage = result.at(gte->operand(0));
      result[inst] = stage;
      result[gte] = stage;
      result[token] = stage;
    }
  }

  // Partition out the Inter IPU copies and also assign stage to their operands.
  auto inter_ipu_copies_end = std::stable_partition(
      outfeeds_end, instructions.end(), util::IsInterIpuCopyInstruction());
  for (auto itr = outfeeds_end; itr != inter_ipu_copies_end; ++itr) {
    HloInstruction* inst = *itr;
    std::queue<HloInstruction*> operands;
    std::vector<HloInstruction*> unassigned_insts;
    operands.push(inst);
    unassigned_insts.push_back(inst);

    int64_t stage = -1;
    while (!operands.empty()) {
      HloInstruction* x = operands.front();
      operands.pop();

      if (result.contains(x)) {
        stage = std::max<int64_t>(result[x], stage);
      } else {
        unassigned_insts.push_back(x);
        for (auto y : x->operands()) {
          operands.push(y);
        }
      }
    }

    CHECK_GT(stage, -1);
    for (auto i : unassigned_insts) {
      result[i] = stage;
    }
  }

  // Partition out GTEs which have not been assigned a stage - these are
  // assigned to the same stage as their input.
  auto gtes_end = std::stable_partition(
      inter_ipu_copies_end, instructions.end(),
      [&result](const HloInstruction* inst) {
        return util::HasHloOpcode(HloOpcode::kGetTupleElement)(inst) &&
               !result.contains(inst);
      });
  for (auto itr = inter_ipu_copies_end; itr != gtes_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_operands(inst);
  }

  // Partition out the copies.
  auto copies_end = std::stable_partition(gtes_end, instructions.end(),
                                          util::HasHloOpcode(HloOpcode::kCopy));
  for (auto itr = gtes_end; itr != copies_end; ++itr) {
    HloInstruction* copy = *itr;
    if (copy->user_count() == 1 && IsResourceUpdate(copy->users()[0])) {
      result[*itr] = get_stage_from_users(*itr);
    } else {
      result[*itr] = get_stage_from_operands(*itr);
    }
  }

  // Partition out FIFOs - if the FIFO is an input to a recomputation stage,
  // then it is assigned to that stage, otherwise it it assigned to the same
  // stage as its input.
  auto fifos_end = std::stable_partition(copies_end, instructions.end(),
                                         util::IsFifoInstruction());
  for (auto itr = copies_end; itr != fifos_end; ++itr) {
    HloInstruction* inst = *itr;
    if (inst->user_count() == 1) {
      if (IsPipelineStageRecomputation(inst->users()[0])) {
        result[inst] = get_stage_from_users(inst);
      } else {
        result[inst] = get_stage_from_operands(inst);
      }
    } else {
      CHECK_EQ(inst->user_count(), 2);
      CHECK(!absl::c_any_of(inst->users(), IsPipelineStageRecomputation));
      result[inst] = get_stage_from_operands(inst);
    }
  }

  // Partition out the gradient accumulation buffers - these are assigned to the
  // first stage in which they are used in.
  auto gradient_accumulators_end =
      std::stable_partition(fifos_end, instructions.end(),
                            util::IsGradientAccumulatorCreateInstruction());
  for (auto itr = fifos_end; itr != gradient_accumulators_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Partition out parameters - these are assigned to the first stage in which
  // they are used in.
  auto parameters_end =
      std::stable_partition(gradient_accumulators_end, instructions.end(),
                            util::HasHloOpcode(HloOpcode::kParameter));
  for (auto itr = gradient_accumulators_end; itr != parameters_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Partition out execution counters - these are assigned to the first stage in
  // which they are used in.
  auto execution_counters_end = std::stable_partition(
      parameters_end, instructions.end(), util::IsExecutionCounter());
  for (auto itr = parameters_end; itr != execution_counters_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Partition out buffers - these are assigned to the first stage in which
  // they are used in.
  auto buffers_end = std::stable_partition(
      execution_counters_end, instructions.end(), util::IsCreateBuffer());
  for (auto itr = parameters_end; itr != buffers_end; ++itr) {
    HloInstruction* inst = *itr;
    result[inst] = get_stage_from_users(inst);
  }

  // Go through the remaining instructions and assign them to stages given their
  // operands. Note that we are visiting in post-order.
  for (auto itr = buffers_end; itr != instructions.end(); ++itr) {
    HloInstruction* inst = *itr;
    // Only assign the stage if no other instruction assigned it for us.
    if (!result.contains(inst)) {
      result[inst] = get_stage_from_operands(inst);
    }
  }

  if (result.size() !=
      static_cast<size_t>(pipeline_computation->instruction_count())) {
    LOG(FATAL) << "Could not assign all the instructions to Pipeline Stages.";
  }
  return result;
}

/**
 * Get the pipeline stages which have stateless recomputation.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The mapping of the ith stage to a IPU device.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
absl::flat_hash_set<int> GetPipelineStagesWithStatelessRecomputation(
    const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stages = GetPipelineStages(pipeline_computation).ValueOrDie();
  absl::flat_hash_set<int> result, tmp;
  absl::c_transform(stages.recomputation, std::inserter(tmp, tmp.begin()),
                    [&stages](const std::pair<int64_t, HloInstruction*>& pair) {
                      // Recomputation stages for stages containing stateful ops
                      // have a different number of operands
                      return (pair.second->operand_count() ==
                              stages.forward[pair.first]->operand_count())
                                 ? pair.first
                                 : -1;
                    });
  // Remove the -1s introduced by the previous transform.
  absl::c_remove_copy(tmp, std::inserter(result, result.begin()), -1);
  return result;
}

/**
 * Get the number of backward stages in the pipeline.
 *
 * @param pipeline The outer pipeline instruction.
 *
 * @returns The number of backward stages in the pipeline.
 *
 * @note Assumes the pipeline is correctly constructed.
 */
int64_t GetNumberOfBackwardPipelineStages(const HloInstruction* pipeline) {
  HloComputation* pipeline_computation = pipeline->to_apply();
  // Cannot reasonably return StatusOr because this is called inside a
  // constructor.
  auto stages = GetPipelineStages(pipeline_computation).ValueOrDie();
  return stages.backward.size();
}

// Return the pipeline stage index for the given hlo instruction
StatusOr<int> GetPipelineStage(
    const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
    const HloInstruction* hlo) {
  if (inst_stage_mapping.count(hlo) == 0) {
    return FailedPrecondition(
        "Hlo instruction \"%s\" does not have an assigned pipeline stage.",
        hlo->ToString());
  }

  return inst_stage_mapping.at(hlo);
}

/**
 * Get all the inputs for the pipeline stage/resource update, making sure to
 * preserve aliasing. Note that there is a mix of inplace and not inplace inputs
 * - we get all of them.
 *
 * @param seq The sequence to use if any copies are inserted.
 * @param res The compiler resources.
 * @param inst The instruction for which we are getting inputs.
 * @param tensor_map The map which stores the tensors.
 *
 * @returns A 2D array of pipeline stage inputs.
 */
StatusOr<TensorOrRemoteBufferVectors> GetInputs(
    DriverProgramSequence& seq, CompilerResources& res,
    const HloInstruction* inst, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TensorOrRemoteBufferVectors inputs(inst->operand_count());
  auto inst_description = GetInplaceDescription(inst);
  // Keep track of inputs which are not inplace (i.e. parameters for forward
  // stages).
  absl::flat_hash_set<int64_t> non_inplace_operand_indices;
  for (int64_t op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
    non_inplace_operand_indices.insert(op_idx);
  }

  // Populate the inputs with the inplace inputs first.
  for (int64_t inplace_idx : inst_description.GetInplaceOperandIndices()) {
    inputs[inplace_idx] = FindInstructionInputs(
        tensor_map, res, inst, inplace_idx, seq, debug_name_and_id, false);
    non_inplace_operand_indices.erase(inplace_idx);
  }
  // Get all the non inplace inputs.
  if (inst_description.GetInplaceOperandIndices().size() !=
      static_cast<size_t>(inst->operand_count())) {
    CHECK(IsAnyPipelineStageOp(inst));
    for (int64_t op_idx : non_inplace_operand_indices) {
      inputs[op_idx] = FindInstructionInputs(tensor_map, res, inst, op_idx, seq,
                                             {debug_name_and_id}, false);
    }
  }
  return inputs;
}
}  // namespace

PipelineVisitor::PipelineVisitor(
    DriverGraph& graph,
    PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule,
    int64_t stage_count, const std::vector<int>& stage_ipu_mapping,
    const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
    const absl::flat_hash_set<int> stages_with_recomputation,
    int64_t num_backward_stages, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id)
    : InplaceDeferredVisitor(res, inputs, description, debug_name_and_id, {}),
      pipeline_scheduler_util_(
          absl::make_unique<util::PipelineSchedulerUtil>(schedule)),
      copy_sequences_(stage_count, {{debug_name_and_id, "copySeq"}}),
      inter_ipu_copy_sequences_(stage_count,
                                {{debug_name_and_id, "interIpuCopySeq"}}),
      fifo_sequences_(stage_count, {{debug_name_and_id, "fifoSeq"}}),
      infeed_sequences_(stage_count, {{debug_name_and_id, "infeedSeq"}}),
      outfeed_sequences_(stage_count, {{debug_name_and_id, "outfeedSeq"}}),
      program_sequences_(stage_count, {{debug_name_and_id, "programSeq"}}),
      recomputation_sequences_(stage_count,
                               {{debug_name_and_id, "recomputationSeq"}}),
      inter_tileset_copy_in_sequences_(
          stage_count, {{debug_name_and_id, "interTilesetCopyInSeq"}}),
      inter_tileset_copy_out_sequences_(
          stage_count, {{debug_name_and_id, "interTilesetCopyOutSeq"}}),
      stage_ipu_mapping_(stage_ipu_mapping),
      inst_stage_mapping_(inst_stage_mapping),
      stages_with_recomputation_(stages_with_recomputation),
      num_backward_stages_(num_backward_stages),
      pipeline_start_sr_method_(GetStochasticRoundingMethod(res)),
      stage_end_sr_methods(stage_count, StochasticRoundingMethod_Undefined) {
  EnterVariableScope();
}

PipelineVisitor::PipelineVisitor(
    DriverGraph& graph, const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id)
    : PipelineVisitor(graph, GetPipelineSchedule(pipeline).ValueOrDie(),
                      GetPipelineStageCount(pipeline),
                      GetPipelineStageDeviceMapping(pipeline),
                      GetPipelineInstStageMapping(pipeline),
                      GetPipelineStagesWithStatelessRecomputation(pipeline),
                      GetNumberOfBackwardPipelineStages(pipeline), res, inputs,
                      description, debug_name_and_id) {
  EnterVariableScope();
}

PipelineVisitor::~PipelineVisitor() = default;

static Status VerifyPipelineArgumentsFixed(int64_t iterations,
                                           int64_t overlap_length) {
  const char* context_message =
      "This number might be called `gradient_accumulation_count`, "
      "`gradient_accumulation_steps_per_replica` or `steps_per_execution` "
      "depending on the API used.";

  if (iterations % overlap_length) {
    // TODO(T11404)
    return FailedPrecondition(
        "The number of iterations of the pipeline must be a multiple of "
        "%d, but it is %d. %s",
        overlap_length, iterations, context_message);
  }
  // To account for ramp up and ramp down we need at least overlap_length
  // iterations.
  if (iterations < overlap_length) {
    return FailedPrecondition(
        "The number of iterations of the pipeline must be at least %d, "
        "but it is %d. %s",
        overlap_length, iterations, context_message);
  }
  return Status::OK();
}

static StatusOr<DriverProgramSequence> VerifyPipelineArgumentsRuntime(
    const HloInstruction* accumulation_count, int64_t overlap_length,
    DriverTensor accumulation_count_tensor, DriverGraph& graph,
    const poplar::DebugContext& debug_context) {
  DriverProgramSequence prog(debug_context);
  auto condition_1 =
      popops::expr::Cast(popops::expr::_1 % overlap_length, poplar::BOOL);
  auto condition_2 = popops::expr::_1 < overlap_length;
  // Incase the counter comes from a different graph get the top graph
  auto counter_graph = graph.getTopLevelGraph();
  auto cond = popops::map(
      counter_graph, std::move(condition_1) || std::move(condition_2),
      {std::move(accumulation_count_tensor)}, prog, debug_context);

  std::string message = absl::StrCat(
      "Pipeline depth is invalid. Check that pipeline depth "
      "is divisible by the overlap length (",
      overlap_length, ") and pipeline depth is >= ", overlap_length,
      ".\nThis number might be called `gradient_accumulation_count`, "
      "`gradient_accumulation_steps_per_replica` or `steps_per_execution` "
      "depending on the API used.");
  prog.add(poplar::program::AbortOnCondition(cond, message, debug_context));
  return prog;
}

StatusOr<DriverProgramSequence> PipelineVisitor::VerifyPipelineArguments(
    const HloInstruction* accumulation_count,
    DriverTensor accumulation_count_tensor, DriverGraph& graph) const {
  const auto iterations = GetAccumulationConstantsValue(accumulation_count);
  const int64_t overlap_length =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_).size();

  if (iterations) {
    TF_RETURN_IF_ERROR(
        VerifyPipelineArgumentsFixed(*iterations, overlap_length));
    return DriverProgramSequence(dnai_);
  }
  return VerifyPipelineArgumentsRuntime(
      accumulation_count, overlap_length, std::move(accumulation_count_tensor),
      graph, {dnai_, "VerifyPipelineConditions"});
}

PipelineVisitor::IterationsType PipelineVisitor::RampDownAdditionalIterations(
    PipelineVisitor::IterationsType iterations, const size_t overlap_length,
    DriverProgramSequence& program) const {
  // Pipelines that verify this is zero can just return 0
  return 0;
}

StatusOr<DriverProgramSequence> PipelineVisitor::GetPipelineSequence(
    IterationsType iterations) const {
  const int64_t overlap_length =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_).size();
  DriverProgramSequence program(dnai_);

  auto ramp_up = GetPipelineRampUpSequence(dnai_);
  auto repeat_block = GetPipelineRepeatBlockSequence(dnai_, iterations);
  auto ramp_down = GetPipelineRampDownSequence(
      dnai_, RampDownAdditionalIterations(iterations, overlap_length, program));

  program.add(pipeline_execution_counters_initialize_sequence_);
  program.add(pipeline_tensors_zeroing_sequence_);
  program.add(pipeline_write_undef_sequence_);
  program.add(ramp_up.program);
  program.add(repeat_block);
  program.add(ramp_down);

  // Add the resource update sequence.
  program.add(resource_update_);

  return program;
}

Status PipelineVisitor::AddSequenceForInstruction(
    const HloInstruction* hlo, const DriverProgramSequence& seq) {
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  switch (hlo->opcode()) {
    case HloOpcode::kCall: {
      if (IsResourceUpdate(hlo)) {
        resource_update_.add(seq);
      } else {
        if (IsPipelineStageRecomputation(hlo)) {
          recomputation_sequences_[stage].add(seq);
        } else {
          program_sequences_[stage].add(seq);
        }
      }
      return Status::OK();
    }
    case HloOpcode::kCopy: {
      if (hlo->user_count() == 1 && IsResourceUpdate(hlo->users()[0])) {
        resource_update_.add(seq);
      } else {
        TF_ASSIGN_OR_RETURN(auto stage,
                            GetPipelineStage(inst_stage_mapping_, hlo));
        copy_sequences_[stage].add(seq);
      }
      return Status::OK();
    }
    case HloOpcode::kGetTupleElement: {
      const HloInstruction* gte_input = hlo->operand(0);
      if (IsResourceUpdate(gte_input)) {
        resource_update_.add(seq);
      } else if (IsPipelineStageRecomputation(gte_input)) {
        recomputation_sequences_[stage].add(seq);
      } else {
        program_sequences_[stage].add(seq);
      }
      return Status::OK();
    }
    case HloOpcode::kInfeed: {
      infeed_sequences_[stage].add(seq);
      return Status::OK();
    }
    case HloOpcode::kParameter: {
      program_sequences_[stage].add(seq);
      return Status::OK();
    }
    case HloOpcode::kTuple: {
      if (hlo->parent()->root_instruction() == hlo) {
        resource_update_.add(seq);
      } else if (util::IsInterTilesetCopyInInstruction(hlo->operand(0))) {
        inter_tileset_copy_in_sequences_[stage].add(seq);
      } else if (util::IsInterTilesetCopyOutInstruction(hlo->operand(0))) {
        inter_tileset_copy_out_sequences_[stage].add(seq);
      } else {
        program_sequences_[stage].add(seq);
      }
      return Status::OK();
    }
    case HloOpcode::kCustomCall: {
      if (util::IsCreateBuffer()(hlo)) {
        pipeline_write_undef_sequence_.add(seq);
        return Status::OK();
      }
      TF_FALLTHROUGH_INTENDED;
    }
    default: {
      return InternalErrorStrCat("Trying to add a sequence for ",
                                 hlo->ToString(), " which is not supported.");
    }
  }
}

Status PipelineVisitor::AppendSequenceGroupedByInstruction(
    const HloInstruction* inst, const DriverProgramSequence& seq) {
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));
  DriverProgramSequence* to_update;
  switch (inst->opcode()) {
    case HloOpcode::kOutfeed: {
      to_update = &outfeed_sequences_[stage];
      break;
    }
    case HloOpcode::kInfeed: {
      to_update = &infeed_sequences_[stage];
      break;
    }
    default: {
      return UnimplementedStrCat(
          "Sequence grouping not implemented in the PipelineVisitor: ",
          inst->ToString());
    }
  }
  to_update->add(seq);
  return Status::OK();
}

Status PipelineVisitor::PrependSequenceGroupedByInstruction(
    const HloInstruction* inst, const DriverProgramSequence& seq) {
  auto& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));
  DriverProgramSequence* to_update;
  switch (inst->opcode()) {
    case HloOpcode::kOutfeed: {
      to_update = &outfeed_sequences_[stage];
      break;
    }
    case HloOpcode::kInfeed: {
      to_update = &infeed_sequences_[stage];
      break;
    }
    default: {
      return UnimplementedStrCat(
          "Sequence grouping not implemented in the PipelineVisitor: ",
          inst->ToString());
    }
  }
  DriverProgramSequence new_sequence({seq, *to_update},
                                     GetDebugNameAndId(inst));
  *to_update = std::move(new_sequence);
  return Status::OK();
}

Status PipelineVisitor::HandleNotImplemented(HloInstruction* hlo) {
  return xla::Unimplemented(
      "%s (%s) is not a valid pipeline stage Hlo instruction",
      hlo->name().c_str(), HloOpcodeString(hlo->opcode()).c_str());
}

/**
 * Creates the PipelineStageVisitor for a PiplineStage or PipelineStageBackward
 * instruction and populates the sequence ready for the execution.
 *
 * @param inst The PiplineStage or PipelineStageBackward instruction which is
 * being lowered.
 *
 * @returns The Poplar sequence with lowering of the stage.
 */
StatusOr<DriverProgramSequence> PipelineVisitor::CreatePipelineStageOp(
    const HloInstruction* inst,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto& graph = GetGraph(resources_, inst);
  DriverProgramSequence seq(debug_name_and_id);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));

  TF_ASSIGN_OR_RETURN(DeferredArgRBVectors inputs,
                      GetInputsForDeferredRBInstruction(inst));

  const bool has_recomputation = stages_with_recomputation_.contains(stage);

  std::unique_ptr<PipelineStageVisitor> visitor;
  if (has_recomputation) {
    DeferredArgRBVectors visitor_inputs = inputs;
    // When recomputation is enabled, we need to add clones for all non read
    // only inputs so that we can reuse the sequence between the forward and the
    // recomputation stage.
    for (int64_t op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
      const HloInstruction* operand = inst->operand(op_idx);
      if (IsPipelineStageReadOnlyInput(operand)) {
        continue;
      }
      for (size_t flat_idx = 0; flat_idx != inputs[op_idx].size(); ++flat_idx) {
        auto optional_tensor = visitor_inputs[op_idx][flat_idx];
        if (optional_tensor) {
          const std::string name =
              absl::StrCat("clone/", op_idx, "/", flat_idx);
          VLOG(3) << "Adding a clone for input (" << op_idx << ", " << flat_idx
                  << ").";
          visitor_inputs[op_idx][flat_idx] = graph.clone(
              optional_tensor->AsTensor(), {debug_name_and_id, name},
              poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        }
      }
    }
    visitor = absl::make_unique<ReusablePipelineStageVisitor>(
        resources_, visitor_inputs, GetInplaceDescription(inst),
        debug_name_and_id);
  } else {
    visitor = absl::make_unique<PipelineStageVisitor>(
        resources_, inputs, GetInplaceDescription(inst), debug_name_and_id);
  }

  HloComputation* stage_computation = inst->to_apply();
  auto order = stage_computation->parent()
                   ->schedule()
                   .sequence(stage_computation)
                   .instructions();

  TF_RETURN_IF_ERROR(stage_computation->AcceptOrdered(visitor.get(), order));
  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(
      visitor->PropagateDeferredAllocations(inst, inputs, debug_name_and_id));

  // Get the sequence for the stage.
  if (has_recomputation) {
    ReusablePipelineStageVisitor* reusable_visitor =
        static_cast<ReusablePipelineStageVisitor*>(visitor.get());

    // Since the sequence is reused, separate execution counters are required to
    // make sure they are correct and independent of the sequence being used for
    // the recomputation stage.
    ExecutionCounters& sequence_counters =
        reusable_visitor->GetExecutionCounters();
    ExecutionCounters forward_counters =
        sequence_counters.Clone(debug_name_and_id);

    // Initialize the counters once.
    pipeline_execution_counters_initialize_sequence_.add(
        forward_counters.SetInitialValuesToZero());

    // Before every execution of the sequence, copy the counters in.
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence counters_in,
        sequence_counters.SetInitialValuesFrom(&forward_counters));
    seq.add(counters_in);

    // Execute the shared sequence.
    seq.add(
        reusable_visitor->GetForwardStageSequence(inst, inputs, tensor_map));

    // After every execution of the sequence, copy the counters out.
    TF_ASSIGN_OR_RETURN(DriverProgramSequence counters_out,
                        sequence_counters.UpdateCounters(&forward_counters));
    seq.add(counters_out);
  } else {
    // Initialize the counters once from the outer scope.
    pipeline_execution_counters_initialize_sequence_.add(
        visitor->GetExecutionCounters().SetInitialValuesToZero());
    // Execute the sequence.
    seq.add(visitor->GetCachedSequence());
  }

  // Set the outputs.
  const TensorOrRemoteBufferVector& pipeline_outputs = visitor->outputs();
  const ShapeTree<bool> add_copies = visitor->GetOutputCopies(inst);
  size_t flat_tuple_index = 0;
  for (const auto& leaf : add_copies.leaves()) {
    auto output = pipeline_outputs[flat_tuple_index];
    if (leaf.second && StageOutputsRequireCopies() && output.IsTensor()) {
      output = DriverTensor(poputil::duplicate(
          graph, output.AsTensor(), seq,
          {debug_name_and_id, absl::StrCat("output/", flat_tuple_index)},
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES));
    }
    TF_CHECK_OK(AddOutput(tensor_map, inst, flat_tuple_index, output));

    flat_tuple_index++;
  }
  CHECK_EQ(pipeline_outputs.size(), flat_tuple_index);

  fwd_stage_visitors_[stage] = std::move(visitor);

  return seq;
}

/**
 * Lowers a PipelineStageRecomputation into Poplar by reusing the sequence from
 * the corresponding PipelineStage visitor if possible, otherwise create a new
 * sequence.
 *
 * @param inst The PipelineStageRecomputation instruction which is being
 * lowered.
 *
 * @returns The Poplar sequence with lowering of the stage.
 */
StatusOr<DriverProgramSequence>
PipelineVisitor::CreatePipelineStageRecomputationOp(
    const HloInstruction* inst,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverProgramSequence seq(debug_name_and_id);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, inst));
  // Get the non-deferred inputs for the pipeline stage.
  TF_ASSIGN_OR_RETURN(auto inputs, GetInputs(seq, resources_, inst, tensor_map,
                                             debug_name_and_id));
  // Recomputation stages reuse the forward stage visitor.
  PipelineStageVisitor* forward_stage_visitor =
      fwd_stage_visitors_.at(stage).get();

  // If the number of inputs of the recomputation doesn't match the number of
  // inputs of the forward stage then this is a stateful recomputation and we
  // need to create a new sequence.
  if (forward_stage_visitor->inputs().size() != inputs.size()) {
    PipelineStageVisitor visitor(
        resources_, ConvertInputsToDeferredInputs(inputs),
        GetInplaceDescription(inst), debug_name_and_id);
    HloComputation* stage_computation = inst->to_apply();
    auto order = stage_computation->parent()
                     ->schedule()
                     .sequence(stage_computation)
                     .instructions();
    TF_RETURN_IF_ERROR(stage_computation->AcceptOrdered(&visitor, order));

    // Initialize the counters once.
    pipeline_execution_counters_initialize_sequence_.add(
        visitor.GetExecutionCounters().SetInitialValuesToZero());

    // Note that it is not required to propagate any deferred allocations here
    // as recomputations do not have any deferred inputs.

    // Get the sequence for the stage.
    seq.add(visitor.GetCachedSequence());

    // Set the outputs.
    const TensorOrRemoteBufferVector& pipeline_outputs = visitor.outputs();
    for (size_t i = 0; i < pipeline_outputs.size(); i++) {
      TF_CHECK_OK(AddOutput(tensor_map, inst, i, pipeline_outputs[i]));
    }
  } else {
    ReusablePipelineStageVisitor* reusable_visitor =
        static_cast<ReusablePipelineStageVisitor*>(forward_stage_visitor);

    // Since the sequence is reused, separate execution counters are required to
    // make sure they are correct and independent of the sequence being used for
    // the forward stage.
    ExecutionCounters& sequence_counters =
        reusable_visitor->GetExecutionCounters();
    ExecutionCounters recomputation_counters =
        sequence_counters.Clone(debug_name_and_id);
    // Initialize the counters once.
    pipeline_execution_counters_initialize_sequence_.add(
        recomputation_counters.SetInitialValuesToZero());

    // Before every execution of the sequence, copy the counters in.
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence counters_in,
        sequence_counters.SetInitialValuesFrom(&recomputation_counters));
    seq.add(counters_in);

    // Execute the shared sequence.
    seq.add(reusable_visitor->GetRecomputationStageSequence(inst, inputs));

    // After every execution of the sequence, copy the counters out.
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence counters_out,
        sequence_counters.UpdateCounters(&recomputation_counters));
    seq.add(counters_out);

    // Set the outputs.
    const TensorOrRemoteBufferVector& pipeline_outputs =
        forward_stage_visitor->outputs();
    for (size_t i = 0; i < pipeline_outputs.size(); i++) {
      TF_CHECK_OK(AddOutput(tensor_map, inst, i, pipeline_outputs[i]));
    }

    // Restore the SR method that was set when first visiting the stage
    // so subsequent calls to MaybeChangeStochasticRoundingMethod produce the
    // same effect. Don't need to do this when doing a full revisit as the
    // the SR mode normally gets set while visiting.
    MaybeSetStochasticRoundingMethod(resources_, stage_end_sr_methods[stage]);
  }
  return seq;
}

Status PipelineVisitor::HandleDeferredAllocationCall(HloInstruction* hlo) {
  HloComputation* comp = hlo->to_apply();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  const std::string sr_change_name = hlo->name() + "_end";

  if (IsResourceUpdate(hlo)) {
    TF_ASSIGN_OR_RETURN(DeferredArgRBVectors inputs,
                        GetInputsForDeferredRBInstruction(hlo));

    // Similar to pipeline stages we always do resource update relative to the
    // starting sr method, since all the pipeline stages do this too then we
    // don't have to worry about how the two get scheduled together.
    MaybeSetStochasticRoundingMethod(resources_, pipeline_start_sr_method_);
    TF_ASSIGN_OR_RETURN(
        DriverProgramSequence seq,
        CreateResourceUpdateOp(resources_, hlo, inputs, hlo->shape(),
                               tensor_map, debug_name_and_id));
    MaybeChangeStochasticRoundingMethod(resources_, sr_change_name,
                                        pipeline_start_sr_method_, seq);

    resource_update_.add(seq);
  } else {
    TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));

    VLOG(1) << "Processing " << hlo->name() << " : " << comp->name()
            << " as a pipeline stage";

    // To increase utilisation pipeline stages may be executed in a non-linear
    // way, instead of keeping track of this order and changing seeds
    // accordingly we always start/end a pipeline stage at the starting sr
    // method. This way we don't have to know about any scheduling specifics.
    MaybeSetStochasticRoundingMethod(resources_, pipeline_start_sr_method_);

    if (IsPipelineStageOrBackwardOp(hlo)) {
      TF_ASSIGN_OR_RETURN(DriverProgramSequence seq,
                          CreatePipelineStageOp(hlo, debug_name_and_id));
      stage_end_sr_methods[stage] = GetStochasticRoundingMethod(resources_);
      MaybeChangeStochasticRoundingMethod(resources_, sr_change_name,
                                          pipeline_start_sr_method_, seq);

      program_sequences_[stage].add(seq);
    } else if (IsPipelineStageRecomputation(hlo)) {
      TF_ASSIGN_OR_RETURN(
          DriverProgramSequence seq,
          CreatePipelineStageRecomputationOp(hlo, debug_name_and_id));
      MaybeChangeStochasticRoundingMethod(resources_, sr_change_name,
                                          pipeline_start_sr_method_, seq);

      recomputation_sequences_[stage].add(seq);
    } else {
      return HandleNotImplemented(hlo);
    }
  }

  return Status::OK();
}

Status PipelineVisitor::HandleNonDeferredCustomCall(HloInstruction* hlo) {
  if (util::IsExecutionCounter()(hlo)) {
    return HandleExecutionCounter(hlo);
  } else if (util::IsFifoInstruction()(hlo)) {
    return HandleFifo(hlo);
  } else if (util::IsInterIpuCopyInstruction()(hlo)) {
    return HandleInterIpuCopy(hlo);
  } else if (util::IsGradientAccumulatorSinkInstruction()(hlo)) {
    return HandleGradientAccumulatorSink(hlo);
  } else if (util::IsInterTilesetCopyInstruction()(hlo)) {
    return HandleInterTilesetCopy(hlo);
  } else {
    return HandleNotImplemented(hlo);
  }
}

Status PipelineVisitor::HandleExecutionCounter(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      CreateCustomCallOp(resources_, hlo, hlo->shape(),
                                         tensor_map, debug_name_and_id));

  program_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleFifo(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      CreateCustomCallOp(resources_, hlo, hlo->shape(),
                                         tensor_map, debug_name_and_id));

  fifo_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleInterIpuCopy(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      CreateCustomCallOp(resources_, hlo, hlo->shape(),
                                         tensor_map, debug_name_and_id));

  inter_ipu_copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleInterTilesetCopy(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      CreateCustomCallOp(resources_, hlo, hlo->shape(),
                                         tensor_map, debug_name_and_id));

  if (util::IsInterTilesetCopyInInstruction(hlo)) {
    inter_tileset_copy_in_sequences_[stage].add(prog);
  } else {
    inter_tileset_copy_out_sequences_[stage].add(prog);
  }

  return Status::OK();
}

Status PipelineVisitor::HandleGradientAccumulatorSink(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  // The sink op just makes sure that all the uses of the gradient
  // accumulation buffer in different pipeline stages on the same device result
  // in the same buffer.
  // No sequence is created.
  TF_RETURN_IF_ERROR(CreateCustomCallOp(resources_, hlo, hlo->shape(),
                                        tensor_map, debug_name_and_id)
                         .status());

  return Status::OK();
}

Status PipelineVisitor::HandleOutfeed(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(hlo);
  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(inst_stage_mapping_, hlo));
  TF_ASSIGN_OR_RETURN(auto progs, CreateOutfeed(resources_, hlo, tensor_map,
                                                debug_name_and_id));

  TF_RETURN_IF_ERROR(
      PrependSequenceGroupedByInstruction(hlo, progs.local_transfer));
  TF_RETURN_IF_ERROR(
      AppendSequenceGroupedByInstruction(hlo, progs.external_transfer));
  return Status::OK();
}

Status PipelineVisitor::HandleDeferredAllocationTuple(HloInstruction* hlo) {
  return InplaceDeferredVisitor::HandleDeferredAllocationTuple(hlo);
}

Status PipelineVisitor::HandleDeferredAllocationWhile(HloInstruction* hlo) {
  return HandleNotImplemented(hlo);
}

Status PipelineVisitor::FinishDeferedAllocationVisit(HloInstruction* inst) {
  DriverGraph& graph = GetMasterGraph(resources_);

  // Create a sequence for all the zeroing gradient accumulation buffers.
  auto& zeroing_tensors =
      resources_.gradient_accumulation_zeroing_tensors.top();
  ZeroTensors(resources_, graph, zeroing_tensors,
              pipeline_tensors_zeroing_sequence_, {dnai_, "ZeroAccumulators"});

  auto& zeroing_remote_buffers =
      resources_.gradient_accumulation_zeroing_remote_buffers.top();
  for (auto zeroing_remote_buffer : zeroing_remote_buffers) {
    pipeline_tensors_zeroing_sequence_.add(zeroing_remote_buffer);
  }

  // Create a sequence for all the write undefs of pipeline tensors (FIFOs).
  auto& write_undefs = resources_.pipelining_write_undef_sequences.top();
  for (DriverProgramSequence& write_undef : write_undefs) {
    pipeline_write_undef_sequence_.add(write_undef);
  }

  TF_RETURN_IF_ERROR(ExitVariableScope());

  // Wrap each of the poplar sequences in a poplar function to maximise code
  // reuse. Transform a given sequence into a poplar function call sequence.
  const poplar::DebugNameAndId& debug_name_and_id = dnai_;
  auto to_function =
      [&graph, debug_name_and_id](
          const DriverProgramSequence& seq) mutable -> DriverProgramSequence {
    auto f = graph.addFunction(seq);
    return DriverProgramSequence({DriverProgramCall(f, {debug_name_and_id})},
                                 {debug_name_and_id});
  };

  // Transform all of the pipeline stage sequences into poplar function calls.
  absl::c_transform(copy_sequences_, copy_sequences_.begin(), to_function);
  absl::c_transform(inter_ipu_copy_sequences_,
                    inter_ipu_copy_sequences_.begin(), to_function);
  absl::c_transform(fifo_sequences_, fifo_sequences_.begin(), to_function);
  absl::c_transform(program_sequences_, program_sequences_.begin(),
                    to_function);
  absl::c_transform(recomputation_sequences_, recomputation_sequences_.begin(),
                    to_function);

  return Status::OK();
}

void PipelineVisitor::AddSequenceForAliasingCopy(
    const HloInstruction*, const DriverProgramSequence& seq) {
  resource_update_.add(seq);
}

std::unique_ptr<PipelineVisitor> ParallelPipelineVisitor::Create(
    DriverGraph& graph, const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return absl::make_unique<ParallelPipelineVisitor>(
      graph, pipeline, res, inputs, description, debug_name_and_id);
}

// Collect the pipeline stage programs and call CreateRampSequences
PipelineVisitor::RepeatBlock ParallelPipelineVisitor::GetPipelineRampUpSequence(
    const poplar::DebugNameAndId& debug_name_and_id) const {
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences =
      util::ConstructRampUpSchedule(offsets, infeed_sequences_, {});
  auto program_sequences =
      util::ConstructRampUpSchedule(offsets, program_sequences_, {});
  auto fifo_sequences =
      util::ConstructRampUpSchedule(offsets, fifo_sequences_, {});
  auto recomputation_sequences = util::ConstructRecomputationRampUpSchedule(
      offsets, recomputation_sequences_, num_backward_stages_, {});
  auto copy_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences = util::ConstructRampUpSchedule(
      offsets, inter_tileset_copy_in_sequences_, {});
  auto inter_tileset_copy_out_sequences = util::ConstructRampUpSchedule(
      offsets, inter_tileset_copy_out_sequences_, {});
  auto outfeed_sequences =
      util::ConstructRampUpSchedule(offsets, outfeed_sequences_, {});

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_in_sequences.begin(),
                          inter_tileset_copy_in_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_out_sequences.begin(),
                          inter_tileset_copy_out_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());

  return {util::DefaultScheduler().CreateRepeatBlock(
              infeed_sequences, debug_name_and_id, offsets.size()),
          offsets.size() / 2};
}

// Collect the pipeline stage programs and call CreateRampSequences
DriverProgramSequence ParallelPipelineVisitor::GetPipelineRampDownSequence(
    const poplar::DebugNameAndId& debug_name_and_id,
    const PipelineVisitor::IterationsType& additional_iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto infeed_sequences = util::ConstructRampDownSchedule(
      offsets, infeed_sequences_, {}, additional_iterations,
      {debug_name_and_id});
  auto program_sequences = util::ConstructRampDownSchedule(
      offsets, program_sequences_, {}, additional_iterations,
      {debug_name_and_id});
  auto fifo_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, fifo_sequences_);
  auto recomputation_sequences = util::ConstructRecomputationRampDownSchedule(
      offsets, recomputation_sequences_, num_backward_stages_, {},
      additional_iterations);
  auto copy_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences = util::ConstructRampDownSchedule(
      offsets, inter_tileset_copy_in_sequences_, {}, additional_iterations,
      {debug_name_and_id});
  auto inter_tileset_copy_out_sequences = util::ConstructRampDownSchedule(
      offsets, inter_tileset_copy_out_sequences_, {}, additional_iterations,
      {debug_name_and_id});
  auto outfeed_sequences = util::ConstructRampDownSchedule(
      offsets, outfeed_sequences_, {}, additional_iterations,
      {debug_name_and_id});

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_in_sequences.begin(),
                          inter_tileset_copy_in_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_out_sequences.begin(),
                          inter_tileset_copy_out_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());
  return util::DefaultScheduler().CreateRepeatBlock(
      infeed_sequences, debug_name_and_id, offsets.size());
}

// Collect the pipeline stage programs and build the repeat block
DriverProgramSequence ParallelPipelineVisitor::GetPipelineRepeatBlockSequence(
    const poplar::DebugNameAndId& debug_name_and_id,
    const IterationsType& iterations) const {
  // Find the set of non-overlapping program offsets.
  std::vector<int> offsets =
      pipeline_scheduler_util_->ScheduleOffsets(stage_ipu_mapping_);

  // Build schedules for the compute and copy programs.
  // Each schedule is 2D, where each column represents a time-slice and each row
  // represents the "mini-batch",
  auto fifo_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, fifo_sequences_);
  auto infeed_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, infeed_sequences_);
  auto program_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, program_sequences_);
  auto recomputation_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, recomputation_sequences_);
  auto copy_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, copy_sequences_);
  auto inter_ipu_copy_sequences = pipeline_scheduler_util_->ConstructSchedule(
      offsets, inter_ipu_copy_sequences_);
  auto inter_tileset_copy_in_sequences =
      pipeline_scheduler_util_->ConstructSchedule(
          offsets, inter_tileset_copy_in_sequences_);
  auto inter_tileset_copy_out_sequences =
      pipeline_scheduler_util_->ConstructSchedule(
          offsets, inter_tileset_copy_out_sequences_);
  auto outfeed_sequences =
      pipeline_scheduler_util_->ConstructSchedule(offsets, outfeed_sequences_);

  // Concatenate the programs in the correct order.
  // We always execute in following order - infeeds, fwd/bwd stages, fifos,
  // recomputation stages, outfeeds and then inter-ipu-copies.
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_in_sequences.begin(),
                          inter_tileset_copy_in_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), program_sequences.begin(),
                          program_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), fifo_sequences.begin(),
                          fifo_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          recomputation_sequences.begin(),
                          recomputation_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_tileset_copy_out_sequences.begin(),
                          inter_tileset_copy_out_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), copy_sequences.begin(),
                          copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(),
                          inter_ipu_copy_sequences.begin(),
                          inter_ipu_copy_sequences.end());
  infeed_sequences.insert(infeed_sequences.end(), outfeed_sequences.begin(),
                          outfeed_sequences.end());

  auto repeat_block = pipeline_scheduler_util_->CreateRepeatBlock(
      infeed_sequences, debug_name_and_id, offsets.size());

  return absl::visit(
      make_visitor<DriverProgramSequence>(
          [&](const int64_t i) -> DriverProgramSequence {
            const int64_t num_repeats = ((i / offsets.size()) - 1);
            if (num_repeats < 1) {
              return DriverProgramSequence(debug_name_and_id);
            }

            return DriverProgramSequence(
                {DriverProgramRepeat(num_repeats, repeat_block,
                                     {debug_name_and_id})},
                {debug_name_and_id});
          },
          [&](const PipelineVisitor::CountAndGraph i) -> DriverProgramSequence {
            DriverProgramSequence result(debug_name_and_id);
            // Incase the counter comes from a different graph get the top graph
            auto counter_graph = i.graph.getTopLevelGraph();
            auto repeat_counter = popops::map(
                counter_graph, (popops::expr::_1 / offsets.size()) - 1,
                {i.count}, result, {debug_name_and_id});
            result.add(popops::countedForLoop(i.graph, 0, repeat_counter, 1,
                                              repeat_block,
                                              {debug_name_and_id}));
            return result;
          }),
      iterations);
}

std::unique_ptr<PipelineVisitor> SequentialPipelineVisitor::Create(
    DriverGraph& graph, const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return absl::make_unique<SequentialPipelineVisitor>(
      graph, pipeline, res, inputs, description, debug_name_and_id);
}

Status SequentialPipelineVisitor::HandleFifo(HloInstruction* hlo) {
  return FailedPrecondition(
      "Fifo operations are not supported in the sequential pipeline");
}

// Collect the pipeline stage programs and call CreateRampSequences
PipelineVisitor::RepeatBlock
SequentialPipelineVisitor::GetPipelineRampUpSequence(
    const poplar::DebugNameAndId& debug_name_and_id) const {
  return {DriverProgramSequence({debug_name_and_id, "RampUp"}), 0};
}

// Collect the pipeline stage programs and call CreateRampSequences
DriverProgramSequence SequentialPipelineVisitor::GetPipelineRampDownSequence(
    const poplar::DebugNameAndId& debug_name_and_id,
    const IterationsType& additional_iterations) const {
  return DriverProgramSequence({debug_name_and_id, "RampDown"});
}

// Collect the pipeline stage programs and build the repeat block
DriverProgramSequence SequentialPipelineVisitor::GetPipelineRepeatBlockSequence(
    const poplar::DebugNameAndId& debug_name_and_id,
    const IterationsType& iterations) const {
  const int64_t num_stages = stage_ipu_mapping_.size();
  // Build a map to execute the recomputation sequence before the backward
  // stage. The recomputation sequences have the same id as the forward stage.
  std::vector<int64_t> recomp_stage_id(num_stages, -1);
  for (int64_t stage_id = 0; stage_id != num_stages; ++stage_id) {
    if (stages_with_recomputation_.contains(stage_id)) {
      CHECK_EQ(num_stages % 2, 0);
      const int64_t num_fwd_stages = num_stages / 2;
      CHECK_LT(stage_id, num_fwd_stages);
      recomp_stage_id[2 * num_fwd_stages + stage_id - 1] = stage_id;
    }
  }

  DriverProgramSequence repeat_block(debug_name_and_id);
  for (int64_t stage_id = 0; stage_id < num_stages; ++stage_id) {
    repeat_block.add(infeed_sequences_[stage_id]);
    repeat_block.add(inter_tileset_copy_in_sequences_[stage_id]);
    if (recomp_stage_id[stage_id] > -1) {
      repeat_block.add(recomputation_sequences_.at(recomp_stage_id[stage_id]));
    }
    repeat_block.add(program_sequences_[stage_id]);
    repeat_block.add(copy_sequences_[stage_id]);
    repeat_block.add(inter_ipu_copy_sequences_[stage_id]);
    repeat_block.add(inter_tileset_copy_out_sequences_[stage_id]);
    repeat_block.add(outfeed_sequences_[stage_id]);
  }

  return util::ForProgram(iterations, std::move(repeat_block),
                          {debug_name_and_id});
}

}  // namespace poplarplugin
}  // namespace xla
