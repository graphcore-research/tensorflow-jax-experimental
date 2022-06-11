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

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_executor.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform_id.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
// #include "tensorflow/stream_executor/lib/status_macros.h"

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

IpuPlatform::IpuPlatform() : name_("IPU") {
  CheckPoplarPackageHash();
  VLOG(tensorflow::INFO) << "Poplar version: " << poplar::versionString()
                         << " Poplar package: " << poplar::packageHash();
}

IpuPlatform::~IpuPlatform() {}

se::Platform::Id IpuPlatform::id() const { return kIpuPlatformId; }

const std::string& IpuPlatform::Name() const { return name_; }

StatusOr<std::unique_ptr<se::StreamExecutor>> IpuPlatform::GetUncachedExecutor(
    const se::StreamExecutorConfig& config) {
  auto executor = absl::make_unique<se::StreamExecutor>(
      this, absl::make_unique<IpuExecutor>(), config.ordinal);
  TF_RETURN_IF_ERROR(executor->Init(config.device_options));

  return std::move(executor);
}

StatusOr<std::unique_ptr<se::DeviceDescription>>
IpuPlatform::DescriptionForDevice(int ordinal) const {
  se::internal::DeviceDescriptionBuilder builder;
  builder.set_name("IPU");
  builder.set_platform_version("");

  return builder.Build();
}

static void InitializeIpuPlatform() {
  std::unique_ptr<se::Platform> platform(new IpuPlatform);
  SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace poplarplugin
}  // namespace xla

REGISTER_MODULE_INITIALIZER(ipu_platform,
                            xla::poplarplugin::InitializeIpuPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(ipu_platform, multi_platform_manager);
