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

#include <mutex>
#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include <vector>

#include "tensorflow/compiler/xla/service/computation_placer.h"
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

  /**
   * @brief IPU device mesh constructor.
   * @param id Id of the mesh.
   * @param ipu_ids Ids of individual IPUs part of the mesh.
   * @param target Poplar target.
   * @param ipu_model_desc Optional IPU model description.
   */
  explicit IpuDeviceMeshInfo(
      IdType id, const std::vector<IdType>& ipu_ids,
      const poplar::Target& target,
      const std::optional<poplar::IPUModel>& ipu_model_desc = std::nullopt);

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
  std::size_t size() const noexcept { return m_ipu_ids.size(); }
  /** IPU target type */
  poplar::TargetType type() const noexcept { return m_target.getTargetType(); }
  /** Is an IPU mesh with single device. */
  bool single() const noexcept { return (m_ipu_ids.size() == 1); }
  /** IPU hardware version. */
  std::string version() const;

  /** Is an IPU part of the mesh? */
  bool isIn(IdType id) const noexcept;
  /** Is an IPU mesh overlapping with another one. */
  bool overlaps(const IpuDeviceMeshInfo& mesh_info) const noexcept;

 private:
  /** Mesh unique id.  */
  IdType m_id;
  /** IPU ids part of the mesh. */
  std::vector<IdType> m_ipu_ids;
  /** Poplar target corresponding to the mesh */
  poplar::Target m_target;
  /** IPU model description, when using an IPU model. */
  std::optional<poplar::IPUModel> m_ipu_model_desc;
};

/**
 * @brief IPU device mesh: combination of mesh info + Poplar device.
 */
class IpuDeviceMesh {
 public:
  using IdType = IpuDeviceMeshInfo::IdType;
  /** Build from Poplar device + child IPU ids. */
  explicit IpuDeviceMesh(
      poplar::Device device, const std::vector<IdType>& child_ipu_ids,
      const std::optional<poplar::IPUModel>& ipu_model_desc = std::nullopt);

  // Standard copy/move.
  IpuDeviceMesh(const IpuDeviceMesh&) = delete;
  IpuDeviceMesh& operator=(const IpuDeviceMesh&) = delete;
  IpuDeviceMesh(IpuDeviceMesh&&) noexcept = default;
  IpuDeviceMesh& operator=(IpuDeviceMesh&&) noexcept = default;

  /** IPU mesh info. */
  const IpuDeviceMeshInfo& info() const noexcept { return m_mesh_info; }
  /** IPU Poplar device. */
  const poplar::Device& device() const { return m_device; }
  /** Is the IPU mesh attached? */
  bool isAttached() const noexcept;

  /** IPU Poplar target corresponding to the mesh. */
  const poplar::Target& target() const noexcept { return m_mesh_info.target(); }
  /** Id of the IPU mesh. */
  IdType id() const noexcept { return m_mesh_info.id(); }
  /** Size of the IPU mesh */
  std::size_t size() const noexcept { return m_mesh_info.size(); }
  /** IPU target type */
  poplar::TargetType type() const noexcept {
    return m_mesh_info.target().getTargetType();
  }
  /** IPU hardware version. */
  std::string version() const noexcept { return m_mesh_info.version(); }
  /** Num tiles per IPU */
  unsigned num_tiles_per_ipu() const noexcept {
    return m_mesh_info.target().getTilesPerIPU();
  }

  /** Is an IPU part of the mesh? */
  bool isIn(IdType id) const noexcept { return m_mesh_info.isIn(id); }
  /** Is an IPU mesh overlapping with another one. */
  bool overlaps(const IpuDeviceMesh& mesh) const noexcept {
    return m_mesh_info.overlaps(mesh.m_mesh_info);
  }

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
  IpuDeviceMeshManager(IpuDeviceMeshManager&&) noexcept;
  IpuDeviceMeshManager& operator=(IpuDeviceMeshManager&&) noexcept = delete;
  IpuDeviceMeshManager(const IpuDeviceMeshManager&) = delete;
  IpuDeviceMeshManager& operator=(const IpuDeviceMeshManager&) = delete;

  /**
   * @brief IPU (local) hardware manager.
   */
  static IpuDeviceMeshManager createIpuManager();
  /**
   * @brief IPU model (simulator) manager
   * @param num_tiles Number of tiles to use in the IPU model.
   * @param version IPU hardware version ('ipu2' or 'ipu21').
   */
  static IpuDeviceMeshManager createIpuModelManager(
      int num_tiles = 4, const std::string& version = "ipu2");

  /** CPU model manager. */
  static IpuDeviceMeshManager createCpuManager();

  /** Is there any local IPU hardware available? */
  static bool hasLocalIpuHardware() noexcept;

  /** Size of the IPU mesh */
  std::size_t size() const noexcept { return m_meshes.size(); }
  /** Type of IPU used in the mesh. */
  poplar::TargetType type() const;

  /** Returns all IPU meshes supported. */
  const std::vector<IpuDeviceMesh>& meshes() const noexcept { return m_meshes; }
  /** Get an IPU mesh at a given index (potentially different from id!). */
  const IpuDeviceMesh& at(std::size_t idx) const;

  /** Get an IPU mesh from its IPU id. */
  const IpuDeviceMesh& find(IdType id) const;
  /** Find an IPU mesh from a list of individual IPU ids. */
  const IpuDeviceMesh& find(std::vector<IdType> ids) const;
  /** Find an IPU mesh from an XLA device assignment. */
  const IpuDeviceMesh& find(const DeviceAssignment& device_assignment) const;

  /** Count the number of meshes with given mesh size. */
  std::size_t count(std::size_t mesh_size = 1) const noexcept;

  /** Get the default mesh for a given number of IPUs required. */
  const IpuDeviceMesh& defaultMesh(std::size_t num_ipus) const;
  /** Convert IPU mesh Id to index in the manager. */
  std::size_t fromMeshIdToIndex(IdType mesh_id) const;
  /** Get overlapping IPU mesh ids. */
  const std::vector<IdType>& overlappingMeshIds(IdType mesh_id) const;

  // Mesh management.
  /** Attach a mesh. */
  bool attach(IdType mesh_id, bool force_detach_overlapping = true) const;
  /** Is a mesh already attached? */
  bool isAttached(IdType mesh_id) const;
  /** Detach a mesh. */
  void detach(IdType mesh_id) const;
  /** Detach all IPU meshes. */
  void detachAll() const;

 private:
  /** Create from a collection of IPU meshes. */
  IpuDeviceMeshManager(std::vector<IpuDeviceMesh> meshes);

  /** IPU meshes available. */
  std::vector<IpuDeviceMesh> m_meshes;
  /** Map caching mesh id to index. */
  std::unordered_map<IdType, std::size_t> m_mesh_it_to_index_map;
  /** Map caching overlapping regions: id => vector of overlapping ids. */
  std::unordered_map<IdType, std::vector<IdType>> m_mesh_overlap_map;

  /** Mutex for any function doing device management (attach, detach, ...) */
  mutable std::mutex m_device_mutex;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
