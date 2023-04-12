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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device_mesh.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_device.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_executable.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

/**
 * @brief IPU PjRt client options.
 */
struct IpuPjRtOptions {
  // NOTE: Visible IPUs field take priority over num_devices if set.
  /** Visible IPUs: by default, all of them (or 1 if using IPU model). */
  std::optional<std::set<int>> visible_devices = std::nullopt;
  /** Number of IPUs to use. By default, all of them. */
  std::optional<int> num_devices = std::nullopt;

  /** Use IPU model. */
  bool use_ipu_model = false;
  /** IPU model number of tiles. */
  int ipu_model_num_tiles = 4;
  /** IPU model hardware version ('ipu2' or 'ipu21') */
  std::string ipu_model_version = "ipu2";

  /** Maximum flops of an XLA computation under which it is directly
   * executed on the host using a CPU client. More specifically:
   * < 0: no executable is run on the HOST;
   * = 0: view only executable run on the HOST;
   * > 0: small programs run on the HOST;
   */
  float execute_on_host_flops_limit = 0.0;

  /* The data which is streamed to/from the device might be stored in different
  layouts on the device and on the host. If so, rearrangement is performed on
  the device by default. By enabling this option the rearrangement will be
  performed on the host at the expense of latency.*/
  bool always_rearrange_copies_on_the_host = true;

  /* If true (default), prefetching of data for data streams on the host will be
    overlapped with execution on the IPU.*/
  bool prefetch_data_streams = true;
};

/**
 * @brief IPU (active) mesh state.
 *
 * This data structure is describing the state of a mesh,
 * with id, executable loaded and latest run id.
 */
struct IpuPjRtMeshState {
 public:
  /** IPU mesh id */
  int mesh_id;
  /** Executable loaded (none by default). */
  int executable_id = 0;
  /** Latest run id (none by default). */
  int run_id = 0;

  /** Latest run buffers reference. Useful for marking on-device as expired.
   * NOTE: updated at every run, to keep track of status of former buffers.
   */
  std::shared_ptr<IpuPjRtRunOutputsRef> run_outputs_ref{nullptr};
  /** Latest run status (PjRt future). */
  std::optional<PjRtFuture<Status>> run_status = std::nullopt;
};

/**
 * @brief IPU PjRt client state: summary of the status of all IPUs.
 *
 * NOTE: as the IPU client is asynchronous, the state represents the last
 * queued operation on the IPU, not necessarily the present state (IPUs
 * attached, ...).
 */
class IpuPjRtClientState {
 public:
  IpuPjRtClientState() = default;

  /**
   * @brief Create initial state, with all individual IPUs (extracted from the
   * IPU mesh manager).
   */
  static IpuPjRtClientState Initialize(
      const IpuDeviceMeshManager& ipu_mesh_manager);
  /**
   * @brief Update a state with executable run info, creating a new instance.
   * Active meshes in the state needs to run on a different mesh configuration.
   */
  IpuPjRtClientState Update(const IpuPjRtExecutableRunInfo& run_info,
                            const IpuDeviceMeshManager& ipu_mesh_manager) const;
  /** Is a mesh active? */
  bool IsActiveMesh(int mesh_id) const;

  /** Find an active mesh by mesh id. */
  const IpuPjRtMeshState* FindByMeshId(int mesh_id) const noexcept;
  /** Find an active mesh by executable id. */
  const IpuPjRtMeshState* FindByExecutableId(int executable_id) const noexcept;

  /** Get client state active meshes. */
  const std::vector<IpuPjRtMeshState>& active_meshes() const {
    return m_active_meshes;
  }
  /** Number of active meshes. */
  std::size_t size() const noexcept { return m_active_meshes.size(); }

 private:
  /** Active/attached IPU meshes. */
  std::vector<IpuPjRtMeshState> m_active_meshes;
};

/**
 * @brief IPU PjRt client.
 */
class IpuPjRtClient : public PjRtClient {
 public:
  /**
   * @brief IPU PjRt client constructor.
   * @param asynchronous Asynchronous client?
   * @param process_id Process id. Should be 0 for single process.
   * @param ipu_mesh_manager IPU device mesh manager.
   * @param devices Collection of single IPU devices.
   * @param options IPU client options.
   */
  explicit IpuPjRtClient(bool asynchronous, int process_id,
                         IpuDeviceMeshManager ipu_mesh_manager,
                         std::vector<IpuPjRtDevice> devices,
                         const IpuPjRtOptions& options);
  virtual ~IpuPjRtClient();

  // Return the process index of this client. Always 0 in single-process
  // settings.
  int process_index() const;

  // Return the number of devices in the entire computation. In multi-headed
  // client setting, some are addressable by this client, some are not. In a
  // single-client setting, this is equal to the number of addressable devices.
  int device_count() const;

  // Return number of addressable devices. Addressable devices are those that
  // the client can issue commands to.
  int addressable_device_count() const;

  // Return all devices known to the client, including addressable and
  // non-addressable devices.
  absl::Span<PjRtDevice* const> devices() const;

  // Return only addressable devices. The devices are in no particular order.
  absl::Span<PjRtDevice* const> addressable_devices() const;

  // Lookup any PjRtDevice for a given PjRtDevice::id().
  virtual StatusOr<PjRtDevice*> LookupDevice(int device_id) const;

  // Return an addressable PjRtDevice for a given
  // PjRtDevice::local_hardware_id().
  virtual StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const;

  // Return an ID that identifies the platform (CPU/GPU/TPU).
  virtual PjRtPlatformId platform_id() const;

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const;

  // Returns an enum that identifies the type of runtime being used under this
  // client.
  virtual PjRtRuntimeType runtime_type() const;

  // Return a device-specific default device assignment, e.g., GPU and TPU may
  // be different.
  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const;

  // Returns a backend-specific HLO cost analysis visitor.
  virtual StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis();

  // Compile `computation` with given `options`.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options);

  // Variant of `Compile` that accepts an MLIR module.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options);

  // Generates a unique fingerprint for `executable`, may be std::nullopt.
  virtual StatusOr<std::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const;

  // Returns a platform-specific serialization of `executable`. The
  // serialization is not guaranteed to be stable over time. `executable` must
  // have been produced by this client.
  virtual StatusOr<std::string> SerializeExecutable(
      const PjRtExecutable& executable) const;

  // Deserializes a serialized executable as produced by
  // SerializeExecutable(). `serialized` must have been produced by a client of
  // the same platform and version as this one.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized, CompileOptions options);

  // Creates a buffer on the device without initializing or copying any data.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device);

  virtual StatusOr<std::unique_ptr<AsyncBufferTransferManager>>
  CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                PjRtDevice* device);

  // on_done_with_host_buffer is optional and may be null.
  // on_done_with_host_buffer will be called iff an OK status is returned.
  //
  // `data` points to the backing array of the host buffer. Caution:
  // `byte_strides` are allowed to be negative, in which case `data` may need
  // to point to the interior of the buffer, not necessarily its start.
  //
  // If byte_strides is omitted, the array is assumed to have a dense layout
  // with dimensions in major-to-minor order.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer, PjRtDevice* device);

  // Note that literal must remain in scope until the transfer has completed, so
  // the caller should, for example, wait for GetReadyFuture().Await()
  // completes on the return value before letting literal go out of scope.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device);

  // Creates a PjRtBuffer that is a non-owned view of an on-device
  // buffer (typically allocated by another library).
  // on_delete_callback is called when the PjRtBuffer is done with the on-device
  // buffer. The buffer may be mutated, for example, if the buffer is donated
  // to an Execute operation.
  // TODO(phawkins): Currently this API assumes the buffer is ready to use
  // immediately on the device. Extend it to support, for example, waiting for a
  // CUDA stream/event.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback);

  // Returns a vector of PjRtBuffers that can be used to receive
  // cross host transfers using `client` on `device'. Asynchronously calls
  // `notifier` once receive descriptors are ready to be communicated to the
  // sender. `shapes` must be the exact shapes, with identical layouts,
  // corresponding to the buffers that will be sent. When resources for the
  // transfer are available, notifier will be called with a vector of
  // PjRtCrossHostRecvDescriptors structs, one for each shape in `shapes`. Each
  // struct contains an opaque string that should be transmitted to the sending
  // host and used in a call to CopyToRemoteDevice. None of the recv buffers
  // will become ready until *all* of the sends have completed.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier);

  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier);

  // Create ChannelHandles for XLA send/recv.
  virtual StatusOr<ChannelHandle> CreateChannelHandle();
  virtual StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle();
  virtual StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle();

  // TODO(zhangqiaorjc): Experimental API to be removed.
  // Defragment device memory.
  Status Defragment();

  // IPU PjRt client specific interface.
  // IPU PjRt client specific interface.
  // IPU PjRt client specific interface.

  /** Get IPU mesh manager. */
  const IpuDeviceMeshManager& ipu_mesh_manager() const noexcept;
  /** Get underlying TFRT CPU client. */
  TfrtCpuClient* cpu_client() const noexcept {
    return tensorflow::down_cast<TfrtCpuClient*>(m_cpu_client.get());
  }
  /** Get underlying TFRT CPU device. */
  TfrtCpuDevice* cpu_device() const noexcept {
    return tensorflow::down_cast<TfrtCpuDevice*>(
        this->cpu_client()->addressable_devices()[0]);
  }

  /** Next IPU executable run id. */
  int64_t next_run_id() { return m_run_id_counter.increment(); }

  /** Get the IPU client state. */

  /**
   * @brief Update the IPU client state with a new executable run.
   * This function is thread-safe, blocking any other modification on the state.
   *
   * It returns: (run_info, previous_state, new_state).
   *
   * TODO: pass the PjRtFuture as well.
   */
  std::tuple<IpuPjRtExecutableRunInfo, IpuPjRtClientState, IpuPjRtClientState>
  UpdateClientState(int mesh_id, int executable_id,
                    std::shared_ptr<IpuPjRtRunOutputsRef> run_outputs_ref);

  /**
   * @brief Should we run an IPU XLA computation directly on HOST? For
   * efficiency and user experience.
   *
   * This function is using HLO module analysis to decide whether to run a
   * computation on host. At the moment, multi devices computation are not
   * supported on host.
   */
  StatusOr<bool> IsIpuExecutableRunOnHost(
      const XlaComputation& computation,
      const CompileOptions& options) const noexcept;

 private:
  bool m_asynchronous;
  /** Process id */
  int m_process_index;
  /** IPU device mesh manager. */
  IpuDeviceMeshManager m_ipu_mesh_manager;
  /** IPU devices, as exposed in PjRt interface. */
  std::vector<IpuPjRtDevice> m_devices;
  /** Vector of pointers to IPU devices. */
  std::vector<PjRtDevice*> m_ptr_devices;
  /** IPU client options. */
  IpuPjRtOptions m_options;

  /** Host/CPU client, to handle buffers on host. */
  std::unique_ptr<PjRtClient> m_cpu_client;

  /** IPU PjRt stream executor. */
  std::unique_ptr<PjRtStreamExecutorClient> m_se_mesh_client;
  /** Stream executor devices: all possible IPU meshes / Poplar devices.
   */
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> m_se_mesh_devices;

  /** Client state. */
  IpuPjRtClientState m_client_state;
  /** Client/state execute mutex. */
  mutable std::mutex m_client_state_mutex;

  /** Global executable id counter. */
  AtomicCounter m_executable_id_counter{1};
  /** Global run id counter. */
  AtomicCounter m_run_id_counter{1};
};

/**
 * @brief Create an IPU device mesh manager from IPU client options.
 * @throw Error in case of invalid/unsupported option.
 */
StatusOr<IpuDeviceMeshManager> CreateIpuDeviceMeshManager(
    const IpuPjRtOptions& ipu_options);

/**
 * @brief Create IPU PjRt devices.
 */
std::vector<IpuPjRtDevice> CreateIpuDevices(
    const IpuDeviceMeshManager& mesh_manager);

/**
 * @brief Build an IPU PjRt client instance.
 * @param asynchronous Asynchronous client.
 * @param ipu_options IPU client options.
 */
StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(
    bool asynchronous, const IpuPjRtOptions& ipu_options);

}  // namespace poplarplugin
}  // namespace xla
