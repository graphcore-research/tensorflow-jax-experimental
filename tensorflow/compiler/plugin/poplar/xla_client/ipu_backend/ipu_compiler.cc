/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/assign_default_layout.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/input_output_aliasing.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/feed_token_verifier.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace poplarplugin {

Status IpuCompiler::RunHloOptimization(HloModule* module) {
  HloPassPipeline pipeline("IpuCompiler");

  pipeline.AddPass<AssignDefaultLayoutIfAbsent>();
  pipeline.AddPass<InputOutputAliasing>();
  pipeline.AddPass<FeedTokenVerifier>();

  return pipeline.Run(module).status();
}

StatusOr<std::unique_ptr<HloModule>> IpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
    const CompileOptions& options) {
  VLOG(1) << "IpuCompiler: Run hlo passes on graph " << module->name();
  TF_RETURN_IF_ERROR(RunHloOptimization(module.get()));
  return std::move(module);
}

se::Platform::Id IpuCompiler::PlatformId() const { return kIpuPlatformId; }

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> IpuCreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool IpuRegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      xla::poplarplugin::kIpuPlatformId, &IpuCreateComputationPlacer);
  return true;
}

bool ipu_placer_registration = IpuRegisterComputationPlacer();

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      xla::poplarplugin::kIpuPlatformId,
      []() { return absl::make_unique<xla::poplarplugin::IpuCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
