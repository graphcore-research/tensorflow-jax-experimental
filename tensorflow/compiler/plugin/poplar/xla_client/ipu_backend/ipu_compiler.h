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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_COMPILER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_COMPILER_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_compiler.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {
namespace poplarplugin {

class IpuCompiler : public PoplarCompiler {
 public:
  IpuCompiler() {}
  ~IpuCompiler() override {}

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) override;

  se::Platform::Id PlatformId() const override;

 private:
  Status RunHloOptimization(HloModule* module);

  TF_DISALLOW_COPY_AND_ASSIGN(IpuCompiler);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_COMPILER_H_
