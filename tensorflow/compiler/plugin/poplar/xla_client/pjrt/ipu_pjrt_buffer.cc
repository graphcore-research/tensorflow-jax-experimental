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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_executable.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/utils.h"

namespace xla {
namespace poplarplugin {

IpuPjRtBufferLocation IpuBufferLocationFromIOType(
    InputOutputAliasingMap::InputInfo::Type iotype) noexcept {
  using InType = InputOutputAliasingMap::InputInfo::Type;
  switch (iotype) {
    case InType::StreamedVariable:
      return IpuPjRtBufferLocation::HOST;
    case InType::ResourceModified:
      return IpuPjRtBufferLocation::SRAM;
    case InType::ResourceNotModified:
      return IpuPjRtBufferLocation::SRAM;
  }
  return IpuPjRtBufferLocation::UNKNOWN;
}
IpuPjRtBufferLocation IpuBufferLocationFromIOType(
    InputOutputAliasingMap::OutputInfo::Type iotype) noexcept {
  using OutType = InputOutputAliasingMap::OutputInfo::Type;
  switch (iotype) {
    case OutType::StreamedVariable:
      return IpuPjRtBufferLocation::HOST;
    case OutType::ResourceModified:
      return IpuPjRtBufferLocation::SRAM;
    case OutType::ResourceOutputOnly:
      return IpuPjRtBufferLocation::SRAM;
  }
  return IpuPjRtBufferLocation::UNKNOWN;
}

IpuPjRtBufferStatus::IpuPjRtBufferStatus(
    IpuPjRtBufferLocation location) noexcept
    : m_location{location},
      m_on_device_expired{false},
      m_is_host_buffer_sync{false} {
  // Host or unknown located buffer => on device expired.
  if (m_location == IpuPjRtBufferLocation::HOST ||
      m_location == IpuPjRtBufferLocation::UNKNOWN) {
    this->MarkOnDeviceExpired();
  }
}

IpuPjRtBuffer::IpuPjRtBuffer() {}
IpuPjRtBuffer::~IpuPjRtBuffer() {
  // Always mark on device expired, in case other buffer tries access it.
  status()->MarkOnDeviceExpired();
}

const Shape& IpuPjRtBuffer::on_device_shape() const {
  // HOST buffer should always have proper shape, equivalent to IPU.
  // TODO: using specific `m_device_shape` in case HOST buffer differ?
  CHECK_NOTNULL(m_host_buffer.get());
  return m_host_buffer->on_device_shape();
}

StatusOr<Shape> IpuPjRtBuffer::logical_on_device_shape() {
  // No dynamic shape currently supported.
  return on_device_shape();
}
PjRtDevice* IpuPjRtBuffer::device() const { return m_device; }
PjRtClient* IpuPjRtBuffer::client() const { return m_device->client(); }

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
IpuPjRtBuffer::AcquireExternalReference() {
  // Already synchronized host buffer.
  if (IsHostBufferSync()) {
    return m_host_buffer->AcquireExternalReference();
  }
  const auto location = this->location();
  if (location != IpuPjRtBufferLocation::SRAM) {
    return Unimplemented(
        "Unsupported IPU buffer location for `AcquireExternalReference`.");
  }
  // Try synchronize SRAM on-device buffer.
  CHECK_NOTNULL(m_run_outputs_ref.get());
  // TODO: first check expire status, or executable ptr.
  TF_RETURN_IF_ERROR(m_run_outputs_ref->executable->CopyDeviceToHostBuffers(
      m_run_outputs_ref.get()));
  MarkHostBufferSynchronized();
  // Now can acquire external reference on host buffer!
  return m_host_buffer->AcquireExternalReference();
}

PjRtFuture<Status> IpuPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  throw std::runtime_error("Not implemented `ToLiteral` on IPU.");
}

StatusOr<size_t> IpuPjRtBuffer::GetOnDeviceSizeInBytes() const {
  // Size on host and IPU should coincide.
  CHECK_NOTNULL(m_host_buffer);
  return m_host_buffer->GetOnDeviceSizeInBytes();
}

PjRtFuture<Status> IpuPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                int64_t transfer_size) {
  // TODO: support on IPU host buffers?
  throw std::runtime_error("Not implemented `CopyRawToHost` on IPU.");
}

void IpuPjRtBuffer::Delete() {
  // Rely on HOST buffer logic to control deleting buffer.
  CHECK_NOTNULL(m_host_buffer);
  m_host_buffer->Delete();
  // Make sure we mark on device expired too.
  status()->MarkOnDeviceExpired();
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
IpuPjRtBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return Unimplemented(
      "Not implemented `ReleaseDeviceMemoryOwnership` on IPU.");
}

bool IpuPjRtBuffer::IsDeleted() {
  CHECK_NOTNULL(m_host_buffer.get());
  const bool is_host_deleted = m_host_buffer->IsDeleted();
  const auto location = this->status()->location();
  if (location == IpuPjRtBufferLocation::HOST) {
    // Rely fully on HOST buffer logic.
    return is_host_deleted;
  } else if (location == IpuPjRtBufferLocation::SRAM) {
    const bool is_on_device_expired = status()->IsOnDeviceExpired();
    // Is buffer deleted: host or on device.
    const bool is_deleted = is_host_deleted || is_on_device_expired;
    // TODO: take host synchronization into account?
    if (is_on_device_expired && !is_host_deleted) {
      // Delete host buffer.
      m_host_buffer->Delete();
    }
    return is_deleted;
  }
  throw std::runtime_error(
      "Not implemented `IsDeleted` on IPU for DRAM buffers.");
}

StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  // Buffer not on the HOST => transfer not supported.
  if (location() != IpuPjRtBufferLocation::HOST) {
    return Unimplemented(
        "`CopyToDevice` not implemented for IPU buffers not located on the "
        "host.");
  }
  // Need host buffer!
  CHECK_NOTNULL(m_host_buffer);
  // NOTE: following CPU client convention. TODO, support?
  if (dst_device == m_device) {
    return InvalidArgument(
        "IPU `CopyToDevice` cannot accept the same source and destination "
        "devices");
  }
  // Not the same client: not yet supported. TODO: use CPU client mechanism
  // copy to device here.
  if (dst_device->client() != m_device->client()) {
    return InvalidArgument(
        "IPU `CopyToDevice` to a different PjRt client is not supported.");
  }
  // Same IPU client, different IPU device. Keep same data on host, wrapped in a
  // new IPU host buffer. Tracked buffer protected by shared_ptr.
  TF_ASSIGN_OR_RETURN(
      auto host_buffer_hold,
      this->GetHostBufferWithHold(TfrtCpuBuffer::ScopedHold::Type::kUsage));
  // No executable if buffer on host.
  return IpuPjRtBuffer::CreateIpuBuffer(
      this->on_device_shape(), host_buffer_hold.buffer(),
      IpuPjRtBufferLocation::HOST, dst_device, nullptr);
}

using RemoteSendCallback =
    std::function<void(Status status, bool sends_were_enqueued)>;
void IpuPjRtBuffer::CopyToRemoteDevice(absl::string_view serialized_descriptor,
                                       RemoteSendCallback on_done) {
  throw std::runtime_error("Not implemented `CopyToRemoteDevice` on IPU.");
}

void IpuPjRtBuffer::CopyToRemoteDeviceScattered(
    absl::Span<const std::pair<std::string, RemoteSendCallback>>
        serialized_descriptors_and_callbacks,
    const ScatterDetails& scatter_details) {
  throw std::runtime_error(
      "Not implemented `CopyToRemoteDeviceScattered` on IPU.");
}

PjRtFuture<Status> IpuPjRtBuffer::GetReadyFuture() {
  // Rely on HOST buffer for async. events / ready future.
  CHECK_NOTNULL(m_host_buffer);
  return m_host_buffer->GetReadyFuture();
}

bool IpuPjRtBuffer::IsOnCpu() const {
  // As we always keep an equivalent host allocated buffer, always considered
  // as being located on CPU. Avoid additional host copy, as Numpy (or other)
  // can directly acquire reference of internal host buffer.
  // See: `AcquireExternalReference` method.
  return true;
}

std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBuffer(
    std::unique_ptr<TfrtCpuBuffer> cpu_buffer, IpuPjRtBufferLocation location,
    PjRtDevice* device, IpuPjRtExecutable* executable) {
  return IpuPjRtBuffer::CreateIpuBuffer(
      std::move(cpu_buffer), std::make_shared<IpuPjRtBufferStatus>(location),
      device, executable);
}
std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBuffer(
    std::unique_ptr<TfrtCpuBuffer> cpu_buffer,
    std::shared_ptr<IpuPjRtBufferStatus> status, PjRtDevice* device,
    IpuPjRtExecutable* executable) {
  const auto location = status->location();
  // A couple of initial checks.
  CHECK_NOTNULL(device);
  CHECK_NOTNULL(cpu_buffer.get());
  CHECK(location != IpuPjRtBufferLocation::UNKNOWN);
  CHECK(location != IpuPjRtBufferLocation::DRAM);
  // Build wrapping IPU buffer.
  std::unique_ptr<IpuPjRtBuffer> buffer = std::make_unique<IpuPjRtBuffer>();
  // Default buffer status (not synced).
  buffer->m_status = std::move(status);
  buffer->m_device = device;
  buffer->m_host_buffer = std::move(cpu_buffer);
  // Keep executable only if not host buffer.
  if (location != IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(executable);
    buffer->m_executable = executable;
  } else {
    buffer->m_executable = nullptr;
  }
  return buffer;
}

std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBuffer(
    const Shape& shape,
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
    IpuPjRtBufferLocation location, PjRtDevice* device,
    IpuPjRtExecutable* executable) {
  return IpuPjRtBuffer::CreateIpuBuffer(
      shape, std::move(tracked_host_buffer),
      std::make_shared<IpuPjRtBufferStatus>(location), device, executable);
}
std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBuffer(
    const Shape& shape,
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
    std::shared_ptr<IpuPjRtBufferStatus> status, PjRtDevice* device,
    IpuPjRtExecutable* executable) {
  // CPU client & device.
  auto ipu_client = tensorflow::down_cast<IpuPjRtClient*>(device->client());
  TfrtCpuClient* cpu_client = ipu_client->cpu_client();
  TfrtCpuDevice* cpu_device = ipu_client->cpu_device();
  // CPU HOST buffer.
  auto host_buffer = std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_host_buffer), cpu_client, cpu_device);
  // IPU buffer, wrapping the host one.
  return IpuPjRtBuffer::CreateIpuBuffer(std::move(host_buffer),
                                        std::move(status), device, executable);
}

StatusOr<TfrtCpuBuffer*> IpuPjRtBuffer::GetHostBuffer(bool allow_unsync) {
  const bool allowed = allow_unsync || IsHostBufferSync();
  if (!allowed) {
    return InvalidArgument(
        "Can not return HOST buffer from an IPU buffer not synchronized with "
        "host.");
  }
  return m_host_buffer.get();
}

StatusOr<TfrtCpuBuffer::ScopedHold> IpuPjRtBuffer::GetHostBufferWithHold(
    TfrtCpuBuffer::ScopedHold::Type type, bool allow_unsync) {
  TF_ASSIGN_OR_RETURN(TfrtCpuBuffer * buffer,
                      this->GetHostBuffer(allow_unsync));
  return buffer->GetBufferWithHold(type);
}

void IpuPjRtBuffer::AsHostOrDelete() {
  if (location() == IpuPjRtBufferLocation::HOST) {
    // Nothing to do!
  } else if (this->IsHostBufferSync()) {
    // Convert to host buffer.
    // TODO: should we wait for events/ready status?
    // m_location = IpuPjRtBufferLocation::HOST;
    throw std::runtime_error("Unsupported");
  } else {
    // Default: not synced, just delete the buffer.
    this->Delete();
  }
}

StatusOr<std::shared_ptr<MaybeOwningCpuMemory>> CreateRawHostBuffer(
    const Shape& shape) {
  if (!shape.IsArray()) {
    return InvalidArgument(
        "Only supporting XLA array shape to create a raw host buffer.");
  }
  const auto size = ShapeUtil::ByteSizeOf(shape);
  return MaybeOwningCpuMemory::AllocateShared(size);
}

}  // namespace poplarplugin
}  // namespace xla
