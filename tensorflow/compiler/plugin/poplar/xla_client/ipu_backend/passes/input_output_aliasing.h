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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_PASSES_INPUT_OUTPUT_ALIASING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_PASSES_INPUT_OUTPUT_ALIASING_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

/**
 * Our Tensorflow updates ExecutableBuildOptions in XlaCompilationCache
 * the ExecutableBuildOptions passes through
 * ExecutionOptions->HloModuleConfig->HloModule so poplar compiler could build
 * InputOutputAliasingMap in CompilerResources JAX jit does not update
 * ExecutableBuildOptions it can only annotate jit donate_argnums to update the
 * input_output_alias_config_ in HloModule
 * this pass use this info to update HloModuleConfig in HloModule
 * thus enables Visitor to deal with resource & resource update
 */
class InputOutputAliasing : public HloModulePass {
 public:
  absl::string_view name() const override { return "input-output-aliasing"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_PASSES_INPUT_OUTPUT_ALIASING_H_
