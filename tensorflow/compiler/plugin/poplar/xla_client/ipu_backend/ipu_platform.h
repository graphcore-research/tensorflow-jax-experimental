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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_PLATFORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_PLATFORM_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace se = stream_executor;

namespace xla {
namespace poplarplugin {

// IpuPlatform is specially used as the backend platform
// of pjrt client, currently used to support Jax framework.
// It inherits from PoplarPlatform, which allows us to make
// some specific adaptations for Jax without affecting the
// logic of the original tensorflow.
class IpuPlatform : public PoplarPlatform {
 public:
  IpuPlatform();
  ~IpuPlatform() override;

  Platform::Id id() const override;

  const std::string& Name() const override;

  StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<se::DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

  /**
   * @brief IPU device count: for JAX PjRt client, count all Poplar devices.
   */
  int VisibleDeviceCount() const;

 private:
  // This platform's name.
  std::string name_;

  SE_DISALLOW_COPY_AND_ASSIGN(IpuPlatform);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_PLATFORM_H_
