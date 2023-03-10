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

#include "tensorflow/compiler/plugin/poplar/driver/passes/replication_factor_to_constant.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

ReplicationFactorToConstant::ReplicationFactorToConstant(
    int32 replication_factor)
    : replication_factor_(replication_factor) {}

StatusOr<bool> ReplicationFactorToConstant::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto comp : module->MakeComputationPostOrder(execution_threads)) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (IsPoplarInstruction(PoplarOp::ReplicationFactor)(inst)) {
        auto replacement = comp->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(replication_factor_)));

        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(replacement));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
