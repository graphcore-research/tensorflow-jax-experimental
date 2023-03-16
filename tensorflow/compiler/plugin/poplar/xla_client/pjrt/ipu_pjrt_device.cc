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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_device.h"

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

IpuPjRtDevice::IpuPjRtDevice(const IpuDeviceMeshInfo& device_info)
    : m_process_index{0}, m_client{nullptr}, m_device_info{device_info} {
  // Must be a single IPU device mesh info.
  CHECK_EQ(m_device_info.size(), 1);
  if (m_device_info.type() == poplar::TargetType::IPU_MODEL) {
    m_device_kind = "Graphcore-IPUModel";
  } else {
    m_device_kind = "Graphcore-IPU";
  }
  m_device_kind = absl::StrFormat("%s(%s)-NumTiles(%i)", m_device_kind,
                                  m_device_info.version(),
                                  m_device_info.target().getTilesPerIPU());
}
IpuPjRtDevice::~IpuPjRtDevice() {}

PjRtClient* IpuPjRtDevice::client() const { return m_client; }
void IpuPjRtDevice::SetClient(PjRtClient* client) {
  m_client = client;
  // Update process id as well.
  m_process_index = client->process_index();
}

bool IpuPjRtDevice::IsAddressable() const {
  // All devices addressable by default.
  return true;
}

int IpuPjRtDevice::id() const {
  // TODO: should be unique in multi-process case.
  return m_device_info.id();
}
int IpuPjRtDevice::process_index() const { return m_process_index; }
int IpuPjRtDevice::local_hardware_id() const {
  // Opaque hardware ID: using IPU Poplar target id.
  return m_device_info.id();
}

absl::string_view IpuPjRtDevice::device_kind() const { return m_device_kind; }

std::string IpuPjRtDevice::DebugString() const {
  // Using default string description for now.
  return ToString();
}
std::string IpuPjRtDevice::ToString() const {
  return absl::StrFormat("IpuDevice(num_tiles=%i, version=%s)",
                         m_device_info.target().getTilesPerIPU(),
                         m_device_info.version());
}

// Returns a scoped event that the caller uses to tell the PjRtClient that
// there is asynchronous work happening that depends on activity on the
// PjRtDevice. See comment on class definition in pjrt_future.h.
//
// Only some PjRtDevice implementations support ScopedAsyncTrackingEvent, and
// those that do not will return nullptr.
std::unique_ptr<ScopedAsyncTrackingEvent>
IpuPjRtDevice::CreateAsyncTrackingEvent(absl::string_view description) const {
  throw std::runtime_error("`CreateAsyncTrackingEvent` not implemented");
}

// Transfer the given literal to the infeed queue.
Status IpuPjRtDevice::TransferToInfeed(const LiteralSlice& literal) {
  throw std::runtime_error(
      "`TransferToInfeed` not implemented on IPU PjRt device.");
}

// Transfer and return a value of the given shape from the outfeed queue.
Status IpuPjRtDevice::TransferFromOutfeed(MutableBorrowingLiteral literal) {
  throw std::runtime_error(
      "`TransferFromOutfeed` not implemented on IPU PjRt device.");
}

// Returns vendor specific attributes about the device. For example the model
// number of a GPU, or the mesh coordinates of a TPU device. The returned
// reference will remain valid for the lifetime of the PjRtDevice.
const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
IpuPjRtDevice::Attributes() const {
  return m_attributes;
}

const IpuDeviceMeshInfo& IpuPjRtDevice::device_info() const noexcept {
  return m_device_info;
}

const IpuDeviceMeshManager& IpuPjRtDevice::ipu_mesh_manager() const noexcept {
  IpuPjRtClient* client = tensorflow::down_cast<IpuPjRtClient*>(m_client);
  return client->ipu_mesh_manager();
}

}  // namespace poplarplugin
}  // namespace xla
