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

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

namespace xla {
namespace poplarplugin {

constexpr char kIpuName[] = "ipu";
static const char kIpuPlatformName[] = "ipu";

namespace {

// A custom PjRtClient for Ipu
class IpuClient : public xla::PjRtStreamExecutorClient {
 public:
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;
};

// Builds an xla::LocalClient for the IPU platform.
StatusOr<LocalClient*> GetIpuXlaClient(const IpuOptions& ipu_options) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("IPU"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible Graphcore IPU devices.");
  }

  LocalClientOptions options;
  options.set_platform(platform);

  size_t device_count = ipu_options.device_config().size();
  CHECK_GT(device_count, 0);

  std::set<int> allowed_devices;
  for (int i = 0; i < device_count; i++) {
    allowed_devices.insert(i);
  }
  options.set_allowed_devices(allowed_devices);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

// Builds a LocalDeviceState for each IPU present.
StatusOr<std::vector<std::unique_ptr<LocalDeviceState>>> BuildLocalDeviceStates(
    LocalClient* xla_client, bool asynchronous, const IpuOptions& ipu_options) {
  std::vector<std::unique_ptr<LocalDeviceState>> local_devices;

  size_t device_count = ipu_options.device_config().size();
  CHECK_GT(device_count, 0);
  for (int i = 0; i < device_count; ++i) {
    se::StreamExecutor* executor =
        xla_client->backend().stream_executor(i).ValueOrDie();

    auto* e = static_cast<PoplarExecutor*>(executor->implementation());
    TF_RETURN_IF_ERROR(e->ConfigurePoplarDevice(ipu_options));

    local_devices.push_back(absl::make_unique<LocalDeviceState>(
        executor, xla_client, LocalDeviceState::kComputeSynchronized,
        asynchronous,
        /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(local_devices);
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& local_device : local_device_states) {
    int device_ordinal = local_device->device_ordinal();
    const se::DeviceDescription& description =
        local_device->executor()->GetDeviceDescription();
    auto device = absl::make_unique<IpuDevice>(
        device_ordinal, std::move(local_device), description.name());
    devices.push_back(std::move(device));
  }
  return devices;
}

StatusOr<IpuOptions> ParseIpuConfig(const IpuConfig& config) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("IPU"));
  size_t num_ipus = config.num_ipus;
  size_t max_num_ipus = platform->VisibleDeviceCount();
  if (num_ipus == 0 || num_ipus > max_num_ipus) {
    return InvalidArgument("Invalid number ipus to attach: ", num_ipus,
                           ", visible device count: ", max_num_ipus, ".");
  }

  IpuOptions options;
  for (size_t i = 0; i < num_ipus; i++) {
    options.add_device_config()->set_auto_count(1);
  }
  return options;
}
}  // namespace

IpuDevice::IpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state,
                     std::string device_kind)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*device_kind=*/std::move(device_kind)) {}

StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(bool asynchronous,
                                                   const IpuConfig& config) {
  IpuOptions options = ParseIpuConfig(config).ValueOrDie();
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client, GetIpuXlaClient(options));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalDeviceState>> local_device_states,
      BuildLocalDeviceStates(xla_client, asynchronous, options));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices = BuildLocalDevices(std::move(local_device_states));
  return std::unique_ptr<PjRtClient>(std::make_unique<PjRtStreamExecutorClient>(
      kIpuName, xla_client, std::move(devices),
      /*node_id=*/0,
      /*allocator=*/nullptr,
      /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr));
}

}  // namespace poplarplugin
}  // namespace xla