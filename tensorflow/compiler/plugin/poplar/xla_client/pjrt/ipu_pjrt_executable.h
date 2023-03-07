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
 * @brief IPU PJRT executable. This class is wrapping a stream-executor
 * executable and takes care of executing the result.
 */
class IpuPjRtExecutable : public PjRtExecutable {
 public:
  explicit IpuPjRtExecutable(
      std::unique_ptr<PjRtStreamExecutorExecutable> se_executable,
      PjRtClient* client);
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

 private:
  /** Underlying stream executor executable. */
  std::unique_ptr<PjRtStreamExecutorExecutable> m_se_executable;

  /** PjRt client which compiled the executable. */
  PjRtClient* m_client;
};

}  // namespace poplarplugin
}  // namespace xla
