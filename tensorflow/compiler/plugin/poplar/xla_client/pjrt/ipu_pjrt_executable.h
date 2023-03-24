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
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

class PoplarExecutable;
class IpuPjRtClient;

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
 * @brief Representing basic info on a single run of an IPU PjRt executable.
 */
struct IpuPjRtExecutableRunInfo {
  /** IPU mesh id on which is executed the program. */
  int mesh_id = 0;
  /** Executable id, assigned by the client. */
  int executable_id = 0;
  /** Run id, assigned by the client. */
  int run_id = 0;
  // TODO: status future.
};

/**
 * @brief IPU PjRt run raw input buffers (and input events) for a single
 * replica.
 */
struct IpuPjRtRunReplicaInputs {
  /** Host/CPU input (scoped) buffers. */
  absl::InlinedVector<TfrtCpuBuffer::ScopedHold, 4> host_buffers;
  /** Definition events from inputs. */
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> host_deps;

  /**
   * @brief Build instance from a span of replica IPU PjRt input buffers.
   */
  static StatusOr<IpuPjRtRunReplicaInputs> CreateFromIpuPjRtBuffers(
      const std::vector<xla::PjRtBuffer*>& inbuffers);

  /** Connect input stream callbacks. TODO: const method. */
  void ConnectStreamCallbacks(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      int replica, poplar::Engine* engine);
};

/**
 * @brief IPU PjRt run raw output buffers for a single replica.
 */
struct IpuPjRtRunReplicaOutputs {
  /** Raw output host buffers. */
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4>
      raw_host_buffers;
  /** Output buffers shape. */
  absl::InlinedVector<Shape, 4> shapes;

  /** Number of outputs. */
  std::size_t size() const { return raw_host_buffers.size(); }

  /** Allocate replica output buffers from IO infos. */
  static StatusOr<IpuPjRtRunReplicaOutputs> AllocateFromOutputInfos(
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos);

  /**
   * @brief Create wrapping IPU PjRt output buffers.
   */
  std::vector<std::unique_ptr<PjRtBuffer>> CreateIpuPjRtBuffers(
      const tfrt::AsyncValueRef<CpuEvent>& execute_event,
      PjRtDevice* ipu_device) const;

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
  /** Base run info (id, ...). */
  IpuPjRtExecutableRunInfo run_info;
  /** All replicas input buffers. Size: num_replicas. */
  std::vector<IpuPjRtRunReplicaInputs> all_inputs;
  /** All replicas outputs buffers. Size: num_replicas. */
  std::vector<IpuPjRtRunReplicaOutputs> all_outputs;

  /** TFRT execute event (use as definition event for output buffers.) */
  tfrt::AsyncValueRef<CpuEvent> execute_event;
  /** Associated PjRt future status. */
  std::optional<PjRtFuture<Status>> future_status;
  /** Custom random seed. TODO: remove from executable? */
  int64_t random_seed = 0;

  /** Number of replicas. */
  std::size_t num_replicas() const {
    CHECK_EQ(all_inputs.size(), all_outputs.size());
    return all_inputs.size();
  }

  /**
   * @brief Create/initialize run state with IO buffers.
   * @param all_input_handles All replicas input buffers.
   * @param output_infos Output infos.
   */
  static StatusOr<IpuPjRtRunState> CreateWithIOBuffers(
      absl::Span<const std::vector<PjRtBuffer*>> all_input_handles,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos);

  /**
   * @brief Connect Poplar stream callbacks to input and output buffers.
   */
  void ConnectStreamCallbacks(
      const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
      const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
      poplar::Engine* engine);

  /**
   * @brief Create wrapping IPU PjRt output buffers.
   */
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
  CreateOutputIpuPjRtBuffers(const tfrt::AsyncValueRef<CpuEvent>& execute_event,
                             std::vector<PjRtDevice*> ipu_devices) const;
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
      int64_t executable_id,
      std::unique_ptr<PjRtStreamExecutorExecutable> ipu_se_executable,
      std::unique_ptr<TfrtCpuExecutable> host_executable,
      const CompileOptions& compile_options, IpuPjRtClient* client);
  virtual ~IpuPjRtExecutable() = default;

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

  /** (Global/unique) executable id. */
  int64_t m_executable_id;
  /** Underlying IPU/device stream executor executable. */
  std::unique_ptr<PjRtStreamExecutorExecutable> m_ipu_se_executable;
  /** (Optional) CPU/host TFRT executable. */
  std::unique_ptr<TfrtCpuExecutable> m_host_executable;
  /** (Original) compilation options of the executable. */
  CompileOptions m_compile_options;

  /** PjRt client which compiled the executable. */
  IpuPjRtClient* m_client;
  /** IPU device assignment (addressable devices). */
  std::vector<PjRtDevice*> m_devices;
  /** Addressable logical device ids. */
  std::vector<PjRtExecutable::LogicalDeviceIds>
      m_addressable_device_logical_ids;
  /** IPU mesh device id. */
  int m_device_mesh_id = -1;
};

}  // namespace poplarplugin
}  // namespace xla
