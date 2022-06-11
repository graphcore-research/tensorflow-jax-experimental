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

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/assign_default_layout.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> AssignDefaultLayoutIfAbsent::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool result = false;

  for (auto* comp : module->computations(execution_threads)) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->instructions()) {
      if (!LayoutUtil::HasLayout(inst->shape())) {
        LayoutUtil::SetToDefaultLayout(inst->mutable_shape());
        result = true;
      }
    }
  }

  return result;
}

}  // namespace poplarplugin
}  // namespace xla
