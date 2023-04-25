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

#include <poplar/Device.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device_mesh.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client_state.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

class PoplarExecutable;
class IpuPjRtClient;
class IpuPjRtExecutable;

/** Get the underlying Poplar executable from an IPU PjRt SE executable. */
PoplarExecutable* GetPoplarExecutable(PjRtStreamExecutorExecutable* executable);

/**
 * @brief Create XLA compile options to pass to the Poplar compiler.
 * This function is doing the translation from PjRt devices to Poplar device.
 */
CompileOptions CreatePoplarCompileOptions(
    const CompileOptions& compile_options,
    const IpuDeviceMeshManager& mesh_manager);

/** Check a Poplar executable is compatible/valid with IPU PjRt backend. */
Status CheckPoplarExecutableValid(PoplarExecutable* poplar_executable,
                                  const CompileOptions& compile_options);

/**
 * @brief IPU (input) buffer donation type.
 *
 * Donated input PjRt buffers are treated differently depending on whether
 * they are updated or left unchanged. In the first case, there are declared
 * as expired/deleted, whereas in the second case they remain valid (allowing
 * an inference loop to run at full speed with weights on SRAM).
 */
enum class IpuPjRtBufferDonationType : int8_t {
  NONE = 0,   // No input donation. Streamed from HOST memory.
  UNCHANGED,  // Donated input buffer unchanged (e.g. inference weights).
  UPDATED     // Donated input buffer updated (e.g. training weights).
};

/**
 * @brief IPU input buffer donation info.
 */
struct IpuPjRtInputOutputDonationInfo {
  /** Input buffer donation type. */
  IpuPjRtBufferDonationType type = IpuPjRtBufferDonationType::NONE;
  /** Input index (donation idx when applicable). */
  int input_index = -1;
  /** Output index (donation idx when applicable). */
  int output_index = -1;

  /** Convert to (readable) string representation. */
  std::string ToString() const noexcept;
};

/**
 * @brief Representation of an IPU executable input/output aliasing info.
 */
class IpuPjRtInputOutputAliasing {
 public:
  IpuPjRtInputOutputAliasing();
  /** Inputs aliasing info. */
  const std::vector<IpuPjRtInputOutputDonationInfo>& inputs() const noexcept {
    return m_inputs_aliasing;
  }
  /** Outputs aliasing info. */
  const std::vector<IpuPjRtInputOutputDonationInfo>& outputs() const noexcept {
    return m_outputs_aliasing;
  }

  /** Create the input-output aliasing info from the base IPU executable (Poplar
   * or HOST). */
  static IpuPjRtInputOutputAliasing CreateFromBaseExecutable(
      PjRtExecutable* base_executable);

  /** Convert to (readable) string representation. */
  std::string ToString() const noexcept;

 private:
  /** Build from inputs aliasing vector. */
  IpuPjRtInputOutputAliasing(
      std::vector<IpuPjRtInputOutputDonationInfo> inputs_aliasing,
      std::size_t num_outputs);

  /** Inputs aliasing info. */
  std::vector<IpuPjRtInputOutputDonationInfo> m_inputs_aliasing;
  /** (Associated) outputs aliasing info. */
  std::vector<IpuPjRtInputOutputDonationInfo> m_outputs_aliasing;
};

/**
 * @brief IPU PjRt run raw input buffers (and input events) for a single
 * replica.
 */
struct IpuPjRtRunReplicaInputs {
  /** Host/CPU input tracked buffers. */
  absl::InlinedVector<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>, 4>
      host_tracked_buffers;
  /** Definition events from inputs. */
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> host_deps;
  /** IPU buffers status. TODO: is it really necessary? Useful for UNCHANGED
   * buffers. */
  std::vector<std::shared_ptr<IpuPjRtBufferStatus>> buffers_status;

  /** Host/CPU input buffers scoped hold. Usage or Donation depending of input
   * type. Temporary for state construction, emptied and converted to usage
   * event once call is queued.
   */
  absl::InlinedVector<TfrtCpuBuffer::ScopedHold, 4> host_buffers_hold;

  /**
   * @brief Build instance from a span of replica IPU PjRt input buffers.
   */
  static StatusOr<IpuPjRtRunReplicaInputs> CreateFromIpuPjRtBuffers(
      const std::vector<xla::PjRtBuffer*>& inbuffers,
      const IpuPjRtInputOutputAliasing& input_output_aliasing);

  /** Connect input stream callbacks. TODO: const method. */
  void ConnectStreamCallbacks(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      int replica, poplar::Engine* engine);
  /** Connect input donated buffers. TODO: const method. */
  void ConnectH2DStreamDonatedBuffers(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      int replica, poplar::Engine* engine);

  /**
   * @brief Convert buffer usage or donation holds to usage event (or confirmed
   * donation).
   */
  void ConvertBufferHold(tfrt::AsyncValueRef<CpuEvent> execute_event);
};

/**
 * @brief IPU PjRt run raw output buffers for a single replica.
 */
struct IpuPjRtRunReplicaOutputs {
  /** Host/CPU output tracked buffers. */
  absl::InlinedVector<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>, 4>
      host_tracked_buffers;
  /** Output buffers shape. */
  absl::InlinedVector<Shape, 4> shapes;

  /** Number of outputs. */
  std::size_t size() const { return host_tracked_buffers.size(); }

  /** Allocate replica output buffers from IO infos. */
  static StatusOr<IpuPjRtRunReplicaOutputs> AllocateFromOutputInfos(
      const tfrt::AsyncValueRef<CpuEvent>& execute_event,
      const IpuPjRtRunReplicaInputs& inputs,
      const IpuPjRtInputOutputAliasing& input_output_aliasing,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos);

  /**
   * @brief Create wrapping IPU PjRt output buffers.
   * @param execute_event Event to use for buffer definition.
   * @param input_buffers_status Input buffers status, for UNCHANGED buffers.
   * @param input_output_aliasing Input/output aliasing info.
   * @param output_infos All outputs info.
   */
  std::vector<std::unique_ptr<PjRtBuffer>> CreateIpuPjRtBuffers(
      const tfrt::AsyncValueRef<CpuEvent>& execute_event,
      const std::vector<std::shared_ptr<IpuPjRtBufferStatus>>&
          input_buffers_status,
      const IpuPjRtInputOutputAliasing& input_output_aliasing,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
      PjRtDevice* ipu_device, IpuPjRtExecutable* executable) const;

  /** Connect output stream callbacks. TODO: const method. */
  void ConnectStreamCallbacks(
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
      int replica, poplar::Engine* engine);
};

/**
 * @brief IPU PjRt executable run state.
 *
 * This state data structure is storing all infos and input/output buffers
 * necessary to do a single IPU engine run.
 */
struct IpuPjRtRunState {
  /** Base run info (ids, ...). */
  IpuPjRtExecutableRunInfo run_info;
  /** Mesh transition info, prior to engine run. */
  IpuPjRtMeshTransition mesh_transition;

  /** All replicas input buffers. Size: num_replicas. */
  std::vector<IpuPjRtRunReplicaInputs> all_inputs;
  /** All replicas outputs buffers. Size: num_replicas. */
  std::vector<IpuPjRtRunReplicaOutputs> all_outputs;

  /** TFRT execute event (use as definition event for output buffers) */
  tfrt::AsyncValueRef<CpuEvent> execute_event;
  /** Custom random seed. TODO: remove from executable? */
  int64_t random_seed = 0;

  /** Location of IPU donated buffer: HOST or SRAM. */
  IpuPjRtBufferLocation inputs_donated_location =
      IpuPjRtBufferLocation::UNKNOWN;

  IpuPjRtRunState() = default;
  ~IpuPjRtRunState();
  // Move only data structure, no copy.
  // Help making sure we are not doing anything wrong!
  IpuPjRtRunState(IpuPjRtRunState&&) noexcept;
  IpuPjRtRunState& operator=(IpuPjRtRunState&&) noexcept;
  IpuPjRtRunState(const IpuPjRtRunState&) = delete;
  IpuPjRtRunState& operator=(const IpuPjRtRunState&) = delete;

  /** Number of replicas. */
  std::size_t num_replicas() const {
    CHECK_EQ(all_inputs.size(), all_outputs.size());
    return all_inputs.size();
  }

  /** Is the run state empty? */
  bool empty() const noexcept {
    return all_inputs.empty() && all_outputs.empty();
  }

  /**
   * @brief Create/initialize run state with IO buffers.
   *
   * @param execute_event Execute event corresponding to the run.
   * @param all_input_handles All replicas input buffers.
   * @param input_output_aliasing In/out aliasing info.
   * @param output_infos Output infos.
   * @return IPU run state with proper IO buffers.
   */
  static StatusOr<IpuPjRtRunState> CreateWithIOBuffers(
      tfrt::AsyncValueRef<CpuEvent> execute_event,
      absl::Span<const std::vector<PjRtBuffer*>> all_input_handles,
      const IpuPjRtInputOutputAliasing& input_output_aliasing,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos);

  /**
   * @brief Connect Poplar stream callbacks to input and output buffers.
   */
  void ConnectStreamCallbacks(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
      poplar::Engine* engine);
  /**
   * @brief Connect input H2D donated buffers (i.e. transfer
   * weights/parameters).
   */
  void ConnectH2DStreamDonatedBuffers(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      poplar::Engine* engine);

  /**
   * @brief Create wrapping IPU PjRt output buffers.
   */
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
  CreateOutputIpuPjRtBuffers(
      const tfrt::AsyncValueRef<CpuEvent>& execute_event,
      const IpuPjRtInputOutputAliasing& input_output_aliasing,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
      std::vector<PjRtDevice*> ipu_devices,
      IpuPjRtExecutable* executable) const;

  /**
   * @brief Convert input buffer hold (usage or donation) to usage event (or
   * confirmed donation). Using run state execute event.
   */
  void ConvertInputBufferHold();
};

/**
 * @brief Reference to all outputs of an IPU executable run.
 *
 * This data structure is useful to share a common reference between
 * all buffer resulting from the same run, in case on-device SRAM buffers
 * are synchronized back with host.
 */
struct IpuPjRtRunOutputsRef {
  // Reference to a buffer + shared ref status.
  struct BufferAndStatusPtrs {
    /** Pointer to buffer (WARNING: may be invalid!) */
    IpuPjRtBuffer* buffer{nullptr};
    /** IPU buffer status (always valid!). */
    std::shared_ptr<IpuPjRtBufferStatus> status{nullptr};
    /** Input/output aliasing (i.e. buffer donation). */
    IpuPjRtInputOutputDonationInfo aliasing;
  };
  /** IPU executable generating the run. */
  // FIXME: what happens if executable is deleted before buffers?
  IpuPjRtExecutable* executable{nullptr};
  // TODO: add PjRt future status?
  /** Output buffers [replicas, num_outputs]. */
  std::vector<std::vector<BufferAndStatusPtrs>> output_buffers;

  /**
   * @brief Create run output ref instance from buffers, and assign
   * it to the collection of buffers.
   */
  static StatusOr<std::shared_ptr<IpuPjRtRunOutputsRef>> CreateAndAssign(
      const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& out_buffers,
      IpuPjRtExecutable* executable);

  /**
   * @brief Mark on-device buffer as expired.
   * @param keep_unchanged_donated_buffers Keep unchanged donated buffers as
   * valid.
   */
  void MarkOnDeviceExpired(bool keep_unchanged_donated_buffers);
  /** Is any on-device buffer expired? */
  bool IsAnyOnDeviceExpired() const noexcept;
};

/**
 * @brief IPU PJRT executable. This class is wrapping a stream-executor
 * executable and takes care of executing the result.
 */
class IpuPjRtExecutable : public PjRtExecutable {
 public:
  /**
   * @brief Build IPU executable from IPU stream-executor executable,
   * and optional CPU/host executable.
   */
  explicit IpuPjRtExecutable(
      bool asynchronous, int64_t executable_id,
      std::unique_ptr<PjRtStreamExecutorExecutable> ipu_se_executable,
      std::unique_ptr<TfrtCpuExecutable> host_executable,
      const CompileOptions& compile_options, IpuPjRtClient* client);
  virtual ~IpuPjRtExecutable();

  virtual PjRtClient* client() const;
  // Unique name for this executable, e.g., HloModule name.
  virtual absl::string_view name() const;
  virtual int num_replicas() const;
  virtual int num_partitions() const;
  virtual int64_t SizeOfGeneratedCodeInBytes() const;

  virtual const DeviceAssignment& device_assignment() const;

  // The replica and partition indices of device_assignment to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may not be the
  // case on multi-host platforms. If there are 4 replicas and 2 partitions on a
  // single host platform, size of addressable_device_logical_ids_ is 4*2 = 8.
  virtual absl::Span<const PjRtExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const;

  // An addressable_device is one which the client can issue commands to.
  // addressable_devices()[i] is the Device to which
  // addressable_device_logical_ids()[i] is assigned.
  virtual absl::Span<PjRtDevice* const> addressable_devices() const;

  // Return an HloModule (optimized) per partition.
  virtual StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const;

  // Executes on devices addressable by the client. Requires executable has a
  // device_assignment and all devices in the device_assignment are addressable
  // by the client.
  //
  // `argument_handles` is `[num_devices, num_args]`.
  //
  // If returned_futures.has_value():
  //   if Execute does not return an error status:
  //     *returned_futures will be resized to be the same length as the return
  //     vector, and each future will become ready once the corresponding device
  //     execute has completed.
  //   else:
  //     *returned_futures is undefined.
  //
  // The caller is *NOT* required to ensure that PjRtExecutable stays alive
  // until futures are ready.
  virtual StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
          const ExecuteOptions& options,
          std::optional<std::vector<PjRtFuture<Status>>>& returned_futures);

  // Execute the assigned replica/partition on a given `device`. Requires
  // executable has a device_assignment, `device` is present in the
  // device_assignment and addressable by the client.
  //
  // If fill_future is true:
  //   if ExecuteSharded does not return an error status:
  //     returned_future will be filled with a future that will become ready
  //     once the execution has completed.
  //    else:
  //     returned_future will not be modified.
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future);

  // Execute on a given `device`. Requires `device` to be addressable by client.
  // Requires executable has exactly 1 replica and 1 partition and no
  // device_assignment (thus portable).
  //
  // If fill_future is true:
  //   if ExecutePortable does not return an error status:
  //     returned_future will be filled with a future that will become ready
  //     once the execution has completed.
  //    else:
  //     returned_future will not be modified.
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future);

  // Asynchronously free resources after the last execution completes.
  virtual void Delete();

  // True if on-device resources associated with the executable are freed.
  virtual bool IsDeleted();

  /// IPU specific APIs.
  /**
   * @brief Copy/synchronize on-device SRAM buffers to HOST.
   * NOTE: this method is blocking/synchronous the main host thread.
   */
  Status CopyDeviceToHostBuffers(IpuPjRtRunOutputsRef* run_outputs_ref);

  /**
   * @brief Synchronous IPU Poplar run.
   * @param run_state Full run state describing run IO + flags.
   */
  void ExecuteDeviceRun(IpuPjRtRunState& run_state);

  /** Get input/output aliasing info. */
  const IpuPjRtInputOutputAliasing& input_output_aliasing() const noexcept {
    return m_input_output_aliasing;
  }

 private:
  friend class IpuPjRtClient;

  /** Get associated Poplar (mesh) device. */
  const poplar::Device& GetPoplarDevice() const;
  /** Should we use the host executable? */
  bool UseHostExecutable() const noexcept;
  /** Get the base PjRt executable used for execution. */
  PjRtExecutable* GetBaseExecutable() const;

  /** Validate input arguments (shape, ...) */
  Status ValidateArgumentHandles(
      absl::Span<PjRtBuffer* const> argument_handles) const;

  /** Mark input buffers as SRAM ones if donated + unchanged. */
  Status MarkUnchangedDonatedBuffersAsSynchronizedOnSRAM(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles) const;

  /**
   * @brief Execute directly on HOST/CPU.
   *
   * This method is just unwrapping/wrapping IPU buffers, and calling the host
   * executable.
   */
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> ExecuteOnHost(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<Status>>>& returned_futures);

  /** Execute loop fucntion, used in the asynchronous case.
   * This method is run in a separate execute thread.
   */
  void ExecuteAsyncLoop();

  /** Asynchronous execution on IPU? */
  bool m_asynchronous_run = false;
  /** (Global/unique) executable id. */
  int64_t m_executable_id;
  /** Underlying IPU/device stream executor executable. */
  std::unique_ptr<PjRtStreamExecutorExecutable> m_ipu_se_executable;
  /** (Optional) CPU/host TFRT executable. */
  std::unique_ptr<TfrtCpuExecutable> m_host_executable;
  /** (Original) compilation options of the executable. */
  CompileOptions m_compile_options;
  /** Input/output aliasing info. */
  IpuPjRtInputOutputAliasing m_input_output_aliasing;

  /** Poplar engine mutex. To be on the safe side for D2H/H2D transfers. */
  mutable std::mutex m_poplar_engine_mutex;

  /** PjRt client which compiled the executable. */
  IpuPjRtClient* m_client;
  /** IPU device assignment (addressable devices). */
  std::vector<PjRtDevice*> m_devices;
  /** Addressable logical device ids. */
  std::vector<PjRtExecutable::LogicalDeviceIds>
      m_addressable_device_logical_ids;
  /** IPU mesh device id. */
  int m_device_mesh_id = -1;

  /** Asynchronous execute thread. */
  std::thread m_execute_thread;
  /** Asynchronous execute queue. */
  ThreadSafeQueue<IpuPjRtRunState> m_execute_queue;
  /** Executable delete status. */
  std::atomic_bool m_executable_is_deleted{false};

  /** Last run output buffers reference.
   * NOTE: this is useful in the case `IpuPjRtExecutable` instance gets deleted
   * before the output buffers => need to mark these as on-device expired.
   */
  std::shared_ptr<IpuPjRtRunOutputsRef> m_last_run_outputs_ref{nullptr};
};

}  // namespace poplarplugin
}  // namespace xla
