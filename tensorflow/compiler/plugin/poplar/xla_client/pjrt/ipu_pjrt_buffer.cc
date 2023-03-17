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

namespace xla {
namespace poplarplugin {

IpuPjRtBuffer::IpuPjRtBuffer() {}
IpuPjRtBuffer::~IpuPjRtBuffer() {}

const Shape& IpuPjRtBuffer::on_device_shape() const {
  // HOST buffer -> use the device shape from it.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->on_device_shape();
  }
  // Other locations: not yet supported.
  throw std::runtime_error("Not implemented `on_device_shape` on IPU.");
}

StatusOr<Shape> IpuPjRtBuffer::logical_on_device_shape() {
  // No dynamic shape currently supported.
  return on_device_shape();
}
PjRtDevice* IpuPjRtBuffer::device() const { return m_device; }
PjRtClient* IpuPjRtBuffer::client() const { return m_device->client(); }

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
IpuPjRtBuffer::AcquireExternalReference() {
  // HOST buffer -> use existing implementation.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->AcquireExternalReference();
  }
  return Unimplemented("Not implemented `AcquireExternalReference` on IPU.");
}

PjRtFuture<Status> IpuPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  throw std::runtime_error("Not implemented `ToLiteral` on IPU.");
}

StatusOr<size_t> IpuPjRtBuffer::GetOnDeviceSizeInBytes() const {
  // HOST buffer -> use existing implementation.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->GetOnDeviceSizeInBytes();
  }
  return Unimplemented("Not implemented `GetOnDeviceSizeInBytes` on IPU.");
}

PjRtFuture<Status> IpuPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                int64_t transfer_size) {
  throw std::runtime_error("Not implemented `CopyRawToHost` on IPU.");
}

void IpuPjRtBuffer::Delete() {
  // HOST buffer -> use existing implementation.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->Delete();
  }
  throw std::runtime_error("Not implemented `Delete` on IPU.");
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
IpuPjRtBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return Unimplemented(
      "Not implemented `ReleaseDeviceMemoryOwnership` on IPU.");
}

bool IpuPjRtBuffer::IsDeleted() {
  // HOST buffer -> use existing implementation.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->IsDeleted();
  }
  throw std::runtime_error("Not implemented `IsDeleted` on IPU.");
}

StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  return Unimplemented("Not implemented `CopyToDevice` on IPU.");
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
  // HOST buffer -> use existing implementation.
  if (m_location == IpuPjRtBufferLocation::HOST) {
    CHECK_NOTNULL(m_buffer);
    return m_buffer->GetReadyFuture();
  }
  // Other locations: not yet supported.
  throw std::runtime_error("Not implemented `GetReadyFuture` on IPU.");
}

bool IpuPjRtBuffer::IsOnCpu() const {
  return m_location == IpuPjRtBufferLocation::HOST;
}

std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBufferOnHost(
    std::unique_ptr<PjRtBuffer> cpu_buffer, PjRtDevice* device) {
  std::unique_ptr<IpuPjRtBuffer> buffer = std::make_unique<IpuPjRtBuffer>();
  CHECK_NOTNULL(device);
  CHECK_NOTNULL(cpu_buffer.get());
  buffer->m_location = IpuPjRtBufferLocation::HOST;
  buffer->m_device = device;
  buffer->m_buffer = std::move(cpu_buffer);
  return buffer;
}

std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBufferOnHost(
    std::unique_ptr<TfrtCpuBuffer> cpu_buffer, PjRtDevice* device) {
  // TODO: refactor to use TfrtCpuBuffer pointer internally.
  return CreateIpuBufferOnHost(
      std::unique_ptr<PjRtBuffer>{cpu_buffer.release()}, device);
}

std::unique_ptr<PjRtBuffer> IpuPjRtBuffer::CreateIpuBufferOnHost(
    const Shape& shape,
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
    PjRtDevice* device) {
  // CPU client & device.
  auto ipu_client = tensorflow::down_cast<IpuPjRtClient*>(device->client());
  TfrtCpuClient* cpu_client = ipu_client->cpu_client();
  TfrtCpuDevice* cpu_device = ipu_client->cpu_device();
  // CPU HOST buffer.
  auto host_buffer = std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_host_buffer), cpu_client, cpu_device);
  // IPU buffer, wrapping the host one.
  return CreateIpuBufferOnHost(std::move(host_buffer), device);
}

StatusOr<TfrtCpuBuffer*> IpuPjRtBuffer::GetHostBuffer() {
  if (m_location != IpuPjRtBufferLocation::HOST) {
    return InvalidArgument(
        "Can not extract host buffer from IPU buffer: location is not HOST.");
  }
  return tensorflow::down_cast<TfrtCpuBuffer*>(m_buffer.get());
}

StatusOr<TfrtCpuBuffer::ScopedHold> IpuPjRtBuffer::GetHostBufferWithHold(
    TfrtCpuBuffer::ScopedHold::Type type) {
  TF_ASSIGN_OR_RETURN(TfrtCpuBuffer * buffer, this->GetHostBuffer());
  return buffer->GetBufferWithHold(type);
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
