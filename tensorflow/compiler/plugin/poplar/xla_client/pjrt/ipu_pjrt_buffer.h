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
#pragma once
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

/**
 * @brief Where is an IPU buffer located?
 */
enum class IpuPjRtBufferLocation : int { UNKNOWN = 0, HOST, SRAM, DRAM };

/**
 * @brief IPU PjRt buffer.
 *
 * An IPU buffer can be located on:
 *  - HOST RAM: typically inputs/outputs before/after IPU streaming;
 *  - IPU SRAM: model parameters;
 *  - IPU DRAM: not yet supported!
 *
 * This class handles the different type of IPU buffers, and the standard
 * PjRt API for them.
 */
class IpuPjRtBuffer : public PjRtBuffer {
 public:
  IpuPjRtBuffer();
  virtual ~IpuPjRtBuffer();

  virtual const Shape& on_device_shape() const;

  // Same as on_device_shape when the shape is static. When the shape is
  // dynamic, it gathers the metadata from the device and returns a static shape
  // representing the logical shape of the data. This approach is identical to
  // how tensorflow and xrt setup the output buffer in the graph.
  //
  // Since this method actually acquires locks and communicate with the device,
  // it does not have the const qualifier, similar to what ToLiteral does.
  virtual StatusOr<Shape> logical_on_device_shape();
  virtual PjRtDevice* device() const;
  virtual PjRtClient* client() const;

  // ExternalReference is a potentially long-lived reference held while a buffer
  // is being shared by an external framework, e.g., NumPy. A client acquires an
  // external reference by calling PjRtBuffer::AcquireExternalReference() and
  // releases it by deleting the ExternalReference. The external framework
  // should not modify the underlying buffer unless it is confident via its own
  // synchronization that modifications do not race with reads from the
  // PjRtBuffer.
  virtual StatusOr<std::unique_ptr<ExternalReference>>
  AcquireExternalReference();

  // Asynchronously copies the buffer's value into `literal`.
  //
  // Return value is a future the caller can use to discover when the copy has
  // completed. The transfer respects the layout of `literal`; to specify a
  // particular layout, set the layout before calling `ToLiteral`.
  virtual PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal);

  // Returns the number of bytes of the buffer storage on the device.
  virtual StatusOr<size_t> GetOnDeviceSizeInBytes() const;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  virtual PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                           int64_t transfer_size);

  // Drops the buffer's reference to its associated device memory, leaving the
  // buffer in an invalid state. The memory will be freed lazily when all async
  // operations using the buffer have completed, according to the allocation
  // semantics of the underlying platform. Delete may briefly block if another
  // thread is in the process of enqueuing an operation on this buffer, but it
  // will never block for a stream operation to complete. If an external
  // framework holds a reference to the TrackedDeviceBuffer via
  // GetBufferWithExternalReference, the memory will not be freed until the
  // external framework drops the reference.
  virtual void Delete();

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but transfers the device
  // memory ownership out via an ExternalReference rather than
  // freeing the device memory, so that another framework can take ownership of
  // it. A return value of nullptr indicates that PjRtBuffer has been
  // deleted. The buffer returned from Release may be safely dropped at any time
  // even if it still has pending async operations. The client should call
  // GetReadyFuture().Await before calling ReleaseDeviceMemoryOwnership with
  // wait_for_operations_to_complete=false, to ensure that the host has
  // synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from ReleaseDeviceMemoryOwnership.
  virtual StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete);

  // True if and only if Delete or Release has previously been called.
  virtual bool IsDeleted();

  // Copies the buffer to device `dst_device`, performing a d2d transfer when
  // `dst_device` is sharing the same Client, and performing a d2h and h2d copy
  // if `dst_device` lives on a different Client.
  // Returns an error if the buffer is already on dst_device.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device);

  // Copies the buffer to the remote device encoded in serialized_descriptor.
  // This call must be preceded by a call to MakeCrossHostReceiveBuffers on the
  // remote host's destination device. MakeCrossHostReceiveBuffers takes an
  // array of shapes to construct the destination buffers, and a callback
  // supplies an array containing both the destination buffers, and a serialized
  // descriptor for each buffer. For each destination buffer there should be a
  // matching call to src->CopyToRemoteDevice on a remote host for a src buffer
  // of the corresponding shape. serialized_descriptor is the string returned by
  // the callback along with the corresponding destination buffer.
  //
  // When the send either completes or fails, on_done will be called. If
  // status is Ok then it is guaranteed that sends_were_enqueued==true.
  // Otherwise, if sends_were_enqueued==false then the sender should contact
  // the receiver out of band to request cancellation of the transfer. If
  // !status.ok() and sends_were_enqueued==true then it is not possible to
  // determine whether the transfer succeeded and the system is in an
  // undefined state. This undefined state almost certainly indicates an
  // unrecoverable hardware error.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  using RemoteSendCallback =
      std::function<void(Status status, bool sends_were_enqueued)>;
  virtual void CopyToRemoteDevice(absl::string_view serialized_descriptor,
                                  RemoteSendCallback on_done);

  virtual void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details);

  // Returns a future that can be used to discover when the data in the
  // PjRtBuffer has been computed, or an error has occurred.
  //
  // If the buffer has been deleted or donated the returned future will
  // immediately hold an error, however if GetReadyFuture() is called before
  // the buffer has been deleted or donated then the returned future will stay
  // valid (will not transition to error as a consequence of buffer deletion)
  // even if the buffer is subsequently donated or deleted.
  virtual PjRtFuture<Status> GetReadyFuture();

  // Whether this buffer is on CPU and thus allows for certain optimizations.
  virtual bool IsOnCpu() const;

  // IPU specific API
  // IPU specific API
  // IPU specific API

  /** Build an IPU host buffer, from a PjRt CPU buffer. */
  static std::unique_ptr<PjRtBuffer> CreateIpuBufferOnHost(
      std::unique_ptr<PjRtBuffer> cpu_buffer, PjRtDevice* device);
  static std::unique_ptr<PjRtBuffer> CreateIpuBufferOnHost(
      std::unique_ptr<TfrtCpuBuffer> cpu_buffer, PjRtDevice* device);

  /** Build an IPU host buffer, from a tracked CPU buffer. */
  static std::unique_ptr<PjRtBuffer> CreateIpuBufferOnHost(
      const Shape& shape,
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
      PjRtDevice* device);

  /** Get the IPU buffer location. */
  IpuPjRtBufferLocation location() const noexcept { return m_location; }

  /** Get the underlying host buffer, when IPU location is host. */
  StatusOr<TfrtCpuBuffer*> GetHostBuffer();
  /** Get a hold on the host buffer, when existing. */
  StatusOr<TfrtCpuBuffer::ScopedHold> GetHostBufferWithHold(
      TfrtCpuBuffer::ScopedHold::Type type);

 private:
  /** Location of the buffer. */
  IpuPjRtBufferLocation m_location{IpuPjRtBufferLocation::UNKNOWN};
  /** Device of the buffer */
  PjRtDevice* m_device{nullptr};

  /** Underlying buffer (when there is one) */
  std::unique_ptr<PjRtBuffer> m_buffer{nullptr};
};

// Helper factory methods.
/**
 * @brief Create a raw HOST memory buffer, based on an array shape.
 */
StatusOr<std::shared_ptr<MaybeOwningCpuMemory>> CreateRawHostBuffer(
    const Shape& shape);

}  // namespace poplarplugin
}  // namespace xla
