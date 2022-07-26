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

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
StatusOr<bool> HasUniqueToken(const HloInstruction* inst) {
  CHECK_NOTNULL(inst);
  if (inst->opcode() == HloOpcode::kInfeed) {
    const auto* token = inst->operand(0);
    CHECK_EQ(token->shape().IsToken(), true);

    if (token->opcode() == HloOpcode::kAfterAll && token->user_count() == 1) {
      return true;
    }
  } else if (inst->opcode() == HloOpcode::kOutfeed) {
    const auto* token = inst->operand(1);
    CHECK_EQ(token->shape().IsToken(), true);

    if (token->opcode() == HloOpcode::kAfterAll && token->user_count() == 1) {
      return true;
    }
  } else {
    return xla::FailedPrecondition(
        "HasUniqueToken expects infeed/outfeed input");
  }
  return false;
}
}  // namespace

StatusOr<bool> FeedTokenVerifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto comp : module->MakeComputationPostOrder(execution_threads)) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kInfeed) {
        if (!HasUniqueToken(inst).ValueOrDie()) {
          auto* token = comp->AddInstruction(HloInstruction::CreateToken());
          inst->ReplaceOperandWith(0, token);
          changed = true;
        }
      } else if (inst->opcode() == HloOpcode::kOutfeed) {
        if (!HasUniqueToken(inst).ValueOrDie()) {
          auto* token = comp->AddInstruction(HloInstruction::CreateToken());
          inst->ReplaceOperandWith(1, token);
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
