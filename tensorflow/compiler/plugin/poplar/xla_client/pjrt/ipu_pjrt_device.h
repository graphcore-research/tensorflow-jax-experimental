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
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {
namespace poplarplugin {

class IpuPjRtClient;

/**
 * @brief IPU PjRt device: 1-to-1 mapping to real IPU hardware (or IPU model).
 */
class IpuPjRtDevice : public PjRtDevice {
 public:
  /**
   * @brief Build IPU PjRt device.
   */
  explicit IpuPjRtDevice(const IpuDeviceMeshInfo& device_info);
  virtual ~IpuPjRtDevice();

  // Return the client that owns this device.
  virtual PjRtClient* client() const;
  void SetClient(PjRtClient* client);

  // Whether client can issue command to this device.
  virtual bool IsAddressable() const;

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  virtual int id() const;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to PjRtClient::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int process_index() const;

  // Opaque hardware ID, e.g., the CUDA device number, useful for identifying
  // which GPU when interacting with non-JAX code. In general, not guaranteed to
  // be dense, and -1 if undefined.
  virtual int local_hardware_id() const;

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view device_kind() const;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual std::string DebugString() const;

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual std::string ToString() const;

  // Returns a scoped event that the caller uses to tell the PjRtClient that
  // there is asynchronous work happening that depends on activity on the
  // PjRtDevice. See comment on class definition in pjrt_future.h.
  //
  // Only some PjRtDevice implementations support ScopedAsyncTrackingEvent, and
  // those that do not will return nullptr.
  virtual std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const;

  // Transfer the given literal to the infeed queue.
  virtual Status TransferToInfeed(const LiteralSlice& literal);

  // Transfer and return a value of the given shape from the outfeed queue.
  virtual Status TransferFromOutfeed(MutableBorrowingLiteral literal);

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const;

  // Returns IPU device info, with full description of underlying hardware.
  const IpuDeviceMeshInfo& device_info() const noexcept;

 private:
  /** Process index. */
  int m_process_index;
  /** Parent IPU client. */
  PjRtClient* m_client;
  /** IPU device (single mesh) info. */
  IpuDeviceMeshInfo m_device_info;
  /** Device kind description. */
  std::string m_device_kind;
  /** Device vendor attributes. */
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> m_attributes;
};

}  // namespace poplarplugin
}  // namespace xla
