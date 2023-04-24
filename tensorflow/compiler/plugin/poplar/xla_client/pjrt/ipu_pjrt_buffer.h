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
#include <atomic>

#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

class IpuPjRtExecutable;
class IpuPjRtRunOutputsRef;

/**
 * @brief Where is an IPU buffer located?
 */
enum class IpuPjRtBufferLocation : int8_t { UNKNOWN = 0, HOST, SRAM, DRAM };

/** Get the IPU buffer location from IO aliasing map type. */
IpuPjRtBufferLocation IpuBufferLocationFromIOType(
    InputOutputAliasingMap::InputInfo::Type iotype) noexcept;
IpuPjRtBufferLocation IpuBufferLocationFromIOType(
    InputOutputAliasingMap::OutputInfo::Type iotype) noexcept;

/**
 * @brief Thread-safe class handling status of (on-device) buffer.
 *
 * On-device buffers (SRAM/DRAM) are deleted/expired when a new engine is
 * loaded or IPUs are re-attached, meaning the IPU client needs to perform
 * proper bookkeeping of past buffers when such event happens.
 *
 * The status of buffer is shared with PjrtClientState using a `shared_ptr`,
 * in order to keep track of which on-device buffer are still valid.
 */
class IpuPjRtBufferStatus {
 public:
  /** Initialize status from buffer location. */
  IpuPjRtBufferStatus(IpuPjRtBufferLocation location) noexcept;

  /**
   * @brief Set/Mark the buffer as on-device expired.
   *
   * If the buffer was previously SRAM synchronized, then it is
   * converted back to a pure HOST buffer.
   *
   * NOTE: Can not undo this operation on buffer status!
   * @return The previous expired status.
   */
  void MarkOnDeviceExpired() noexcept {
    std::scoped_lock l(m_mutex);
    if (m_is_host_buffer_sync) {
      // Convert back to HOST only buffer when possible.
      // e.g. typical case of re-loading weights in inference.
      m_location = IpuPjRtBufferLocation::HOST;
    }
    m_is_host_buffer_sync = false;
    m_on_device_expired = true;
  }
  /**
   * @brief Convert to SRAM synchronized buffer.
   */
  void ConvertToSRAMHostSynchronized() noexcept {
    std::scoped_lock l(m_mutex);
    // TODO: additional checks?
    m_location = IpuPjRtBufferLocation::SRAM;
    m_on_device_expired = false;
    m_is_host_buffer_sync = true;
  }
  // TODO: removed, replaced with higher level semantics.
  void MarkHostBufferSynchronized() noexcept {
    std::scoped_lock l(m_mutex);
    m_is_host_buffer_sync = true;
  }

  /** IPU buffer location. */
  IpuPjRtBufferLocation location() const noexcept {
    std::scoped_lock l(m_mutex);
    return m_location;
  }
  /** Is the buffer on-device expired? */
  bool IsOnDeviceExpired() const noexcept {
    std::scoped_lock l(m_mutex);
    return m_on_device_expired;
  }
  /** Is the buffer on HOST synchronized with device. Always true for host
   * buffers. */
  bool IsHostBufferSync() const noexcept {
    std::scoped_lock l(m_mutex);
    // Either already on the host, or already synchronized.
    return (m_location == IpuPjRtBufferLocation::HOST) ||
           (m_location != IpuPjRtBufferLocation::UNKNOWN &&
            m_is_host_buffer_sync);
    return true;
  }

 private:
  // Mutex protecting the status (potentially shared between threads).
  // TODO: can use only atomic values?
  mutable std::mutex m_mutex;
  // Location of the buffer. TODO: atomic value?
  IpuPjRtBufferLocation m_location;
  // Is the on-device (SRAM/DRAM) buffer expired (either deleted, or
  // re-written).
  bool m_on_device_expired = false;
  // Is host buffer sync, when buffer is located on SRAM (or DRAM).
  // By default, false (safer option!).
  bool m_is_host_buffer_sync = false;
};

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

  /**
   * @brief Build an IPU HOST or SRAM buffer, with a PjRt host buffer.
   */
  static std::unique_ptr<PjRtBuffer> CreateIpuBuffer(
      std::unique_ptr<TfrtCpuBuffer> cpu_buffer, IpuPjRtBufferLocation location,
      PjRtDevice* device, IpuPjRtExecutable* executable = nullptr);
  static std::unique_ptr<PjRtBuffer> CreateIpuBuffer(
      std::unique_ptr<TfrtCpuBuffer> cpu_buffer,
      std::shared_ptr<IpuPjRtBufferStatus> status, PjRtDevice* device,
      IpuPjRtExecutable* executable = nullptr);

  /** Build an IPU HOST or SRAM buffer, from a tracked CPU buffer. */
  static std::unique_ptr<PjRtBuffer> CreateIpuBuffer(
      const Shape& shape,
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
      IpuPjRtBufferLocation location, PjRtDevice* device,
      IpuPjRtExecutable* executable = nullptr);
  static std::unique_ptr<PjRtBuffer> CreateIpuBuffer(
      const Shape& shape,
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_host_buffer,
      std::shared_ptr<IpuPjRtBufferStatus> status, PjRtDevice* device,
      IpuPjRtExecutable* executable = nullptr);

  /** Get the IPU buffer location. */
  IpuPjRtBufferLocation location() const noexcept {
    return m_status->location();
  }
  /** Get (shareable) buffer status */
  std::shared_ptr<IpuPjRtBufferStatus> status() const noexcept {
    return m_status;
  }

  /** Get the underlying host buffer, when IPU location is host. */
  StatusOr<TfrtCpuBuffer*> GetHostBuffer(bool allow_unsync = false);
  /** Get a hold on the host buffer, when existing (or synced). */
  StatusOr<TfrtCpuBuffer::ScopedHold> GetHostBufferWithHold(
      TfrtCpuBuffer::ScopedHold::Type type, bool allow_unsync = false);

  /**
   * Is the internal host buffer in sync with (optional) IPU SRAM/DRAM buffer.
   * Always true for HOST buffers.
   */
  bool IsHostBufferSync() const noexcept {
    return m_status->IsHostBufferSync();
  }
  /** Mark the IPU buffer as host synchronized. */
  void MarkHostBufferSynchronized() noexcept {
    m_status->MarkHostBufferSynchronized();
  }
  /** Convert the buffer to an SRAM buffer synchronized with host. */
  void ConvertToSRAMHostSynchronized() noexcept {
    m_status->ConvertToSRAMHostSynchronized();
  }

  /**
   * @brief Convert to HOST IPU buffer if synchronized, or delete
   * if not synchronized (and not already on the host).
   */
  void AsHostOrDelete();

  /** Assign run outputs reference. */
  void AssignRunOutputsRef(
      std::shared_ptr<IpuPjRtRunOutputsRef> outputs_ref) noexcept {
    m_run_outputs_ref = std::move(outputs_ref);
  }
  /** Get run outputs reference. */
  std::shared_ptr<IpuPjRtRunOutputsRef> run_outputs_ref() const noexcept {
    return m_run_outputs_ref;
  }

 private:
  /**
   * IPU buffer (shareable) status. A buffer status can be shared with:
   * - IPU client state, to keep track of valid/invalid SRAM buffers;
   * - Other IPU buffers, in the case of `unchanged` donated IPU SRAM buffers;
   *   In the later case, multiple buffers are referencing the same SRAM
   * read-only memory space, hence why it is safe to do so!
   */
  std::shared_ptr<IpuPjRtBufferStatus> m_status{nullptr};
  /** Device of the buffer */
  PjRtDevice* m_device{nullptr};
  /** (Optional) IPU executable generating the buffer. */
  IpuPjRtExecutable* m_executable{nullptr};

  /**
   * Underlying HOST/CPU TFRT buffer.
   *
   * The host buffer is holding an (optional) host copy of data, as well
   * as handling all buffer events. In this way, we re-using a lot of the
   * CPU client logic for asynchronous execution and events.
   */
  std::unique_ptr<TfrtCpuBuffer> m_host_buffer{nullptr};
  /** Reference to output buffers when generated by executable run. */
  std::shared_ptr<IpuPjRtRunOutputsRef> m_run_outputs_ref{nullptr};
};

// Helper factory methods.
/**
 * @brief Create a raw HOST memory buffer, based on an array shape.
 */
StatusOr<std::shared_ptr<MaybeOwningCpuMemory>> CreateRawHostBuffer(
    const Shape& shape);

}  // namespace poplarplugin
}  // namespace xla
