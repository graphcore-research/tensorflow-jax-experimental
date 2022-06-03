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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

class IpuDevice : public PjRtStreamExecutorDevice {
 public:
  IpuDevice(int id, std::unique_ptr<LocalDeviceState> local_device_state,
            std::string device_kind);
};

struct IpuConfig {
  size_t num_ipus = 1;
};

StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(
    bool asynchronous, const IpuConfig& config);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_