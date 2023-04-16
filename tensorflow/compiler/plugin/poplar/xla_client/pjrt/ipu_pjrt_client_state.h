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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace poplarplugin {

class IpuPjRtRunOutputsRef;

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
  /** Execute event of the latest mesh run. */
  tfrt::AsyncValueRef<CpuEvent> execute_event;
};

/**
 * @brief IPU mesh state transition.
 *
 * This data structure is describing how to handle the state
 * transition of an IPU mesh: does it require attaching, loading,
 * which mesh events to wait for?
 */
struct IpuPjRtMeshTransition {
  /** IPU mesh id. */
  int mesh_id = 0;
  /** Does the mesh requires Poplar device to be attached? */
  bool require_device_attach = false;
  /** Does the mesh requires the Poplar engine to be loaded? */
  bool require_engine_load = false;
  /** Collection of mesh blocking events. */
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> mesh_blocking_events;
};

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

  /** (Optional) reference to run outputs */
  std::shared_ptr<IpuPjRtRunOutputsRef> outputs_ref{nullptr};
  /** Run execute event. */
  tfrt::AsyncValueRef<CpuEvent> execute_event;
};

/**
 * @brief IPU PjRt client state: summary of the status of all IPUs.
 *
 * NOTE: as the IPU client is asynchronous, the state represents the last
 * queued operation on the IPU, not necessarily the present state (IPUs
 * attached, ...).
 *
 * Having an explicit IPU client state is helping managing multiple IPUs
 * in a simple way, as the all collection of IPU devices is just considered
 * as a state machine, transitioning from one state to another.
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
   *
   * @return Pair of new IPU client state and IPU mesh transition.
   */
  std::pair<IpuPjRtClientState, IpuPjRtMeshTransition> Update(
      const IpuPjRtExecutableRunInfo& run_info,
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

}  // namespace poplarplugin
}  // namespace xla
