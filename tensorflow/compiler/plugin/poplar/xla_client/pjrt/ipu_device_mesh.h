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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_MESH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_MESH_H_

#include <poplar/Device.hpp>
#include <poplar/Target.hpp>
#include <vector>

#include "tensorflow/compiler/xla/service/platform_util.h"

namespace xla {
namespace poplarplugin {

/**
 * @brief IPU device mesh info.
 *
 * An IPU device mesh corresponds to a partition of IPUs, supported by Poplar.
 * It has unique ID, and a full Poplar target description.
 */
class IpuDeviceMeshInfo {
 public:
  using IdType = unsigned;

  IpuDeviceMeshInfo(IdType id, const std::vector<IdType>& ipu_ids,
                    const poplar::Target& target);

  // Standard copy/move.
  IpuDeviceMeshInfo(const IpuDeviceMeshInfo&) = default;
  IpuDeviceMeshInfo& operator=(const IpuDeviceMeshInfo&) = default;
  IpuDeviceMeshInfo(IpuDeviceMeshInfo&&) noexcept = default;
  IpuDeviceMeshInfo& operator=(IpuDeviceMeshInfo&&) noexcept = default;

  /** Id of the IPU mesh. */
  IdType id() const noexcept { return m_id; }
  /** IPU ids in the mesh. */
  const std::vector<IdType>& ipuIds() const noexcept { return m_ipu_ids; }
  /** IPU Poplar target corresponding to the mesh. */
  const poplar::Target& target() const noexcept { return m_target; }

  /** Size of the IPU mesh */
  size_t size() const noexcept { return m_ipu_ids.size(); }
  /** Is an IPU mesh with single device. */
  bool single() const noexcept { return (m_ipu_ids.size() == 1); }

 private:
  /** Mesh unique id.  */
  IdType m_id;
  /** IPU ids part of the mesh. */
  std::vector<IdType> m_ipu_ids;
  /** Poplar target corresponding to the mesh */
  poplar::Target m_target;
};

/**
 * @brief IPU device mesh: combination of mesh info + Poplar device.
 */
class IpuDeviceMesh {
 public:
  using IdType = IpuDeviceMeshInfo::IdType;
  /** Build from Poplar device + child IPU ids. */
  IpuDeviceMesh(poplar::Device device,
                const std::vector<IdType>& child_ipu_ids);

  // Standard copy/move.
  IpuDeviceMesh(const IpuDeviceMesh&) = delete;
  IpuDeviceMesh& operator=(const IpuDeviceMesh&) = delete;
  IpuDeviceMesh(IpuDeviceMesh&&) noexcept = default;
  IpuDeviceMesh& operator=(IpuDeviceMesh&&) noexcept = default;

  /** IPU mesh info. */
  const IpuDeviceMeshInfo& info() const noexcept { return m_mesh_info; }
  /** IPU Poplar device. */
  const poplar::Device& device() const { return m_device; }

  /** IPU Poplar target corresponding to the mesh. */
  const poplar::Target& target() const noexcept { return m_mesh_info.target(); }
  /** Id of the IPU mesh. */
  IdType id() const noexcept { return m_mesh_info.id(); }
  /** Size of the IPU mesh */
  size_t size() const noexcept { return m_mesh_info.size(); }

 private:
  /** IPU mesh info. */
  IpuDeviceMeshInfo m_mesh_info;
  /** Poplar device. */
  poplar::Device m_device;
};

/**
 * @brief IPU device mesh manager. Providing IPU meshes available.
 */
class IpuDeviceMeshManager {
 public:
  using IdType = IpuDeviceMeshInfo::IdType;
  /** Empty IPU manager. */
  IpuDeviceMeshManager() {}

  // Standard move. No copy.
  IpuDeviceMeshManager(const IpuDeviceMeshManager&) = delete;
  IpuDeviceMeshManager& operator=(const IpuDeviceMeshManager&) = delete;
  IpuDeviceMeshManager(IpuDeviceMeshManager&&) noexcept = default;
  IpuDeviceMeshManager& operator=(IpuDeviceMeshManager&&) noexcept = default;

  /** IPU (local) hardware manager. */
  static IpuDeviceMeshManager createIpuManager();
  /** IPU model (simulator) manager */
  static IpuDeviceMeshManager createIpuModelManager();
  /** CPU model manager. */
  static IpuDeviceMeshManager createCpuManager();

  /** Is there any local IPU hardware available? */
  static bool hasLocalIpuHardware() noexcept;

  /** Size of the IPU mesh */
  size_t size() const noexcept { return m_meshes.size(); }

  /** IPU meshes supported. */
  const std::vector<IpuDeviceMesh>& meshes() const noexcept { return m_meshes; }
  /** Get an IPU mesh from id. */
  const IpuDeviceMesh& mesh(IdType id) const;
  /** Find an IPU mesh from the list of IPU ids. */
  const IpuDeviceMesh& find(std::vector<IdType> ids) const;

 private:
  IpuDeviceMeshManager(std::vector<IpuDeviceMesh> meshes);
  /** IPU meshes available. */
  std::vector<IpuDeviceMesh> m_meshes;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
