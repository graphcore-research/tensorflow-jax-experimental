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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_blocked_recompute_suggestions.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> RemoveBlockedRecomputeSuggestions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloCustomCallInstruction*> custom_calls;

  auto is_suggestion = IsPoplarInstruction(PoplarOp::SuggestRecompute);
  auto is_block = IsPoplarInstruction(PoplarOp::BlockRecompute);

  for (auto comp : module->MakeComputationPostOrder(execution_threads)) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kCustomCall) {
        auto custom_call = Cast<HloCustomCallInstruction>(inst);

        // Have we found a recomputation suggestion?
        if (is_suggestion(custom_call)) {
          auto operand = custom_call->mutable_operand(0);

          // If that suggestion is blocked, remove it.
          if (is_block(operand)) {
            TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(operand));
            TF_RETURN_IF_ERROR(comp->RemoveInstruction(custom_call));

            return true;
          }
        }
      }
    }
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
