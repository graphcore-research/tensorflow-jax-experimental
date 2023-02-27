/* Copyright (c) 2023 Graphcore Ltd. All rights reserved.

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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"

#include <poplar/Graph.hpp>

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace poplarplugin {

constexpr char kIpuName[] = "ipu";
static const char kIpuPlatformName[] = "ipu";

IpuPjRtClient::IpuPjRtClient(bool asynchronous, int process_id,
                             IpuDeviceMeshManager ipu_mesh_manager,
                             std::vector<IpuPjRtDevice> devices)
    : m_asynchronous{asynchronous},
      m_process_index{process_id},
      m_ipu_mesh_manager{std::move(ipu_mesh_manager)},
      m_devices{std::move(devices)} {
  m_ptr_devices.reserve(m_devices.size());
  for (auto& c : m_devices) {
    // Set client pointer in all local IPU devices.
    c.SetClient(this);
    // Internal device pointer array.
    m_ptr_devices.push_back(&c);
  }
  // Tfrt CPU client, handling buffers on host side.
  m_cpu_client = GetTfrtCpuClient(asynchronous, 1).value();
}
IpuPjRtClient::~IpuPjRtClient() {}

int IpuPjRtClient::process_index() const { return m_process_index; }
int IpuPjRtClient::device_count() const { return m_devices.size(); }
int IpuPjRtClient::addressable_device_count() const { return m_devices.size(); }

absl::Span<PjRtDevice* const> IpuPjRtClient::devices() const {
  return m_ptr_devices;
}
absl::Span<PjRtDevice* const> IpuPjRtClient::addressable_devices() const {
  return m_ptr_devices;
}

StatusOr<PjRtDevice*> IpuPjRtClient::LookupDevice(int device_id) const {
  for (auto* ptr_device : m_ptr_devices) {
    if (device_id == ptr_device->id()) {
      return ptr_device;
    }
  }
  return InvalidArgument("No matching IPU device found for `device_id`: %d",
                         device_id);
}
StatusOr<PjRtDevice*> IpuPjRtClient::LookupAddressableDevice(
    int local_hardware_id) const {
  for (auto* ptr_device : m_ptr_devices) {
    if (local_hardware_id == ptr_device->local_hardware_id()) {
      return ptr_device;
    }
  }
  return InvalidArgument(
      "No matching IPU device found for local_hardware_id %d",
      local_hardware_id);
}

PjRtPlatformId IpuPjRtClient::platform_id() const {
  static const PjRtPlatformId kIpuId = tensorflow::Fingerprint64(kIpuName);
  return kIpuId;
}
absl::string_view IpuPjRtClient::platform_name() const {
  return kIpuPlatformName;
}
absl::string_view IpuPjRtClient::platform_version() const {
  // Use poplar::packageHash?
  static const std::string platform_version =
      absl::StrFormat("%s_sdk%s", kIpuPlatformName, poplar::versionString());
  return platform_version;
}
PjRtRuntimeType IpuPjRtClient::runtime_type() const { return kStreamExecutor; }

StatusOr<DeviceAssignment> IpuPjRtClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  // TODO: default IPU mesh?
  return Unimplemented("Not implemented `GetDefaultDeviceAssignment` on IPU.");
}
StatusOr<std::unique_ptr<HloCostAnalysis>> IpuPjRtClient::GetHloCostAnalysis() {
  // TODO: re-direct to StreamExecutor?
  return Unimplemented("Not implemented `GetHloCostAnalysis` on IPU.");
}

StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  return Unimplemented("Not implemented `Compile` on IPU.");
}
StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  // TODO: convert back to XLA.
  return Unimplemented("Not implemented `Compile` on IPU.");
}

// Generates a unique fingerprint for `executable`, may be std::nullopt.
StatusOr<std::optional<std::string>> IpuPjRtClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
  // TODO?
  return Unimplemented("Not implemented `ExecutableFingerprint` on IPU.");
}

StatusOr<std::string> IpuPjRtClient::SerializeExecutable(
    const PjRtExecutable& executable) const {
  return Unimplemented("Not implemented `SerializeExecutable` on IPU.");
}
// Deserializes a serialized executable as produced by
// SerializeExecutable(). `serialized` must have been produced by a client of
// the same platform and version as this one.
StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::DeserializeExecutable(
    absl::string_view serialized, CompileOptions options) {
  return Unimplemented("Not implemented `DeserializeExecutable` on IPU.");
}

// Creates a buffer on the device without initializing or copying any data.
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device) {
  return Unimplemented("Not implemented `CreateUninitializedBuffer` on IPU.");
}
StatusOr<std::unique_ptr<PjRtClient::AsyncBufferTransferManager>>
IpuPjRtClient::CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                             PjRtDevice* device) {
  return Unimplemented(
      "Not implemented `CreateBuffersForAsyncTransfer` on IPU.");
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  // Create IPU buffer on the HOST. Will be transfered on IPU when required.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> cpu_buffer,
                      m_cpu_client->BufferFromHostBuffer(
                          data, type, dims, byte_strides, host_buffer_semantics,
                          std::move(on_done_with_host_buffer), nullptr));
  return IpuPjRtBuffer::createIpuBufferOnHost(std::move(cpu_buffer), device);
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtDevice* device) {
  return Unimplemented("Not implemented `BufferFromHostLiteral` on IPU.");
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback) {
  return Unimplemented("Not implemented `CreateViewOfDeviceBuffer` on IPU.");
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtClient::MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                                           PjRtDevice* device,
                                           PjRtCrossHostRecvNotifier notifier) {
  // Not necessary on single process?
  return Unimplemented("Not implemented `MakeCrossHostReceiveBuffers` on IPU.");
}
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtClient::MakeCrossHostReceiveBuffersForGather(
    absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
    PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) {
  // Not necessary on single process?
  return Unimplemented(
      "Not implemented `MakeCrossHostReceiveBuffersForGather` on IPU.");
}

StatusOr<ChannelHandle> IpuPjRtClient::CreateChannelHandle() {
  return Unimplemented("Not implemented `CreateChannelHandle` on IPU.");
}
StatusOr<ChannelHandle> IpuPjRtClient::CreateDeviceToHostChannelHandle() {
  return Unimplemented(
      "Not implemented `CreateDeviceToHostChannelHandle` on IPU.");
}
StatusOr<ChannelHandle> IpuPjRtClient::CreateHostToDeviceChannelHandle() {
  return Unimplemented(
      "Not implemented `CreateHostToDeviceChannelHandle` on IPU.");
}

Status IpuPjRtClient::Defragment() {
  return Unimplemented("Not implemented `Defragment` on IPU.");
}

// Factory methods.
IpuDeviceMeshManager CreateIpuDeviceMeshManager(
    const IpuPjRtOptions& ipu_options) {
  // Visible devices option not yet supported.
  CHECK(!ipu_options.visible_devices.has_value());
  if (ipu_options.use_ipu_model) {
    return IpuDeviceMeshManager::createIpuModelManager(
        ipu_options.ipu_model_num_tiles, ipu_options.ipu_model_version);
  }
  return IpuDeviceMeshManager::createIpuManager();
}

std::vector<IpuPjRtDevice> CreateIpuDevices(
    const IpuDeviceMeshManager& mesh_manager) {
  std::vector<IpuPjRtDevice> devices;
  for (const auto& m : mesh_manager.meshes()) {
    // Single IPU device mesh only.
    if (m.size() == 1) {
      devices.push_back(IpuPjRtDevice(m.info()));
    }
  }
  return devices;
}

StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(
    bool asynchronous, const IpuPjRtOptions& ipu_options) {
  // Default local process id?
  int process_id = 0;
  auto mesh_manager = CreateIpuDeviceMeshManager(ipu_options);
  auto devices = CreateIpuDevices(mesh_manager);

  return std::unique_ptr<PjRtClient>(std::make_unique<IpuPjRtClient>(
      asynchronous, process_id, std::move(mesh_manager), std::move(devices)));
}

}  // namespace poplarplugin
}  // namespace xla
