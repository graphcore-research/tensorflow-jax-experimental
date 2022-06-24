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

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/input_output_aliasing.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"

#include <vector>

namespace xla {
namespace poplarplugin {

StatusOr<bool> InputOutputAliasing::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto config = module->config();
  CHECK_EQ(config.argument_input_indices().size(), 0);
  CHECK_EQ(config.resource_input_indices().size(), 0);
  CHECK_EQ(config.resource_input_initialized().size(), 0);
  CHECK_EQ(config.resource_update_to_input_index().size(), 0);

  const auto entry = module->entry_computation();
  auto is_tuple = [](const HloInstruction* inst) -> bool {
    return inst->shape().IsTuple() ? true : false;
  };
  if (absl::c_any_of(entry->parameter_instructions(), is_tuple)) {
    return xla::FailedPrecondition(
        "InputOutputAliasing expects non-tuple entry parameter");
  }

  const auto n_parameters = entry->num_parameters();
  std::vector<int32> parameter_indices;
  std::vector<int32> argument_input_indices;
  std::vector<bool> resource_input_initialized;
  std::vector<int32> resource_update_to_input_index;

  // Set resource update through io_alias_config
  module->input_output_alias_config().ForEachAlias(
    [&](const ShapeIndex& output_index, const HloInputOutputAliasConfig::Alias& alias) {
      // Require non-tuple parameter
      CHECK(alias.parameter_index.empty());
      resource_update_to_input_index.push_back(alias.parameter_number);
    }
  );
  config.set_resource_update_to_input_index(resource_update_to_input_index);

  // Set resource update as resource
  config.set_resource_input_indices(resource_update_to_input_index);

  // Set all resource as initialized
  resource_input_initialized.resize(resource_update_to_input_index.size(), true);
  config.set_resource_input_initialized(resource_input_initialized);

  // Set the reset parameters as argument
  parameter_indices.resize(n_parameters);
  absl::c_iota(parameter_indices, 0);
  argument_input_indices.reserve(n_parameters - resource_input_initialized.size());
  absl::c_set_difference(parameter_indices, resource_update_to_input_index,
                         std::back_inserter(argument_input_indices));
  config.set_argument_input_indices(argument_input_indices);

  // Update config without changing other elements
  module->set_config(config);

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
