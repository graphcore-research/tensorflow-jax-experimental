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

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device_mesh.h"

#include <algorithm>
#include <poplar/DeviceManager.hpp>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/default/logging.h"

namespace xla {
namespace poplarplugin {
using IdType = IpuDeviceMeshInfo::IdType;

IpuDeviceMeshInfo::IpuDeviceMeshInfo(
    IdType id, const std::vector<IdType>& ipu_ids, const poplar::Target& target,
    const std::optional<poplar::IPUModel>& ipu_model_desc)
    : m_id{id},
      m_ipu_ids{ipu_ids},
      m_target{target},
      m_ipu_model_desc{ipu_model_desc} {
  // Single IPU case.
  if (m_ipu_ids.size() == 0) {
    m_ipu_ids.push_back(m_id);
  }
  if (m_ipu_ids.size() == 1) {
    CHECK_EQ(m_id, m_ipu_ids[0]);
  }
  // Always sort IPU ids.
  std::sort(m_ipu_ids.begin(), m_ipu_ids.end());
  // Consistent inputs.
  CHECK_GT(m_ipu_ids.size(), 0);
  CHECK_EQ(m_ipu_ids.size(), m_target.getNumIPUs());
}

std::string IpuDeviceMeshInfo::version() const {
  // IPU model: target does not support version query.
  if (m_target.getTargetType() == poplar::TargetType::IPU_MODEL) {
    return m_ipu_model_desc->IPUVersion;
  }
  return m_target.getTargetArchString();
}

IpuDeviceMesh::IpuDeviceMesh(
    poplar::Device device, const std::vector<IdType>& child_ipu_ids,
    const std::optional<poplar::IPUModel>& ipu_model_desc)
    : m_mesh_info{device.getId(), child_ipu_ids, device.getTarget(),
                  ipu_model_desc},
      m_device{std::move(device)} {}

IpuDeviceMeshManager::IpuDeviceMeshManager(std::vector<IpuDeviceMesh> meshes)
    : m_meshes{std::move(meshes)} {}

IpuDeviceMeshManager IpuDeviceMeshManager::createCpuManager() {
  // not yet supported.
  throw std::runtime_error("IPU device mesh `CPU` devices not yet supported");
  return IpuDeviceMeshManager();
}

IpuDeviceMeshManager IpuDeviceMeshManager::createIpuModelManager(
    int num_tiles, const std::string& version) {
  // IPU model description to create.
  poplar::IPUModel ipu_model_desc(version.c_str());
  ipu_model_desc.tilesPerIPU = num_tiles;
  // Only support single IPU for now. TODO: multi-ipus.
  std::vector<IpuDeviceMesh> meshes;
  meshes.push_back(
      IpuDeviceMesh(ipu_model_desc.createDevice(/*optionFlags=*/{},
                                                /*accurateHalf=*/false,
                                                /*deviceManagerId=*/0),
                    {}, ipu_model_desc));
  return IpuDeviceMeshManager(std::move(meshes));
}

IpuDeviceMeshManager IpuDeviceMeshManager::createIpuManager() {
  auto poplar_manager = poplar::DeviceManager::createDeviceManager();
  auto poplar_devices = poplar_manager.getDevices();

  std::vector<IpuDeviceMesh> meshes;
  for (auto& d : poplar_devices) {
    const auto& target = d.getTarget();
    // Ignore non real IPU hardware.
    if (target.getTargetType() != poplar::TargetType::IPU) {
      continue;
    }
    auto child_device_ids = poplar_manager.getChildDeviceIds(d.getId());
    meshes.push_back(IpuDeviceMesh(std::move(d), child_device_ids));
  }
  return IpuDeviceMeshManager(std::move(meshes));
}

bool IpuDeviceMeshManager::hasLocalIpuHardware() noexcept {
  const auto poplar_manager = poplar::DeviceManager::createDeviceManager();
  const auto devices = poplar_manager.getDevices(poplar::TargetType::IPU, 1);
  return !devices.empty();
}

poplar::TargetType IpuDeviceMeshManager::type() const {
  // If empty => throw an error.
  if (m_meshes.empty()) {
    throw std::runtime_error("No IPU device registered in the IPU manager.");
  }
  return m_meshes[0].type();
}

const IpuDeviceMesh& IpuDeviceMeshManager::at(std::size_t idx) const {
  return m_meshes.at(idx);
}

const IpuDeviceMesh& IpuDeviceMeshManager::find(IdType id) const {
  auto it = std::find_if(m_meshes.begin(), m_meshes.end(),
                         [id](const IpuDeviceMesh& m) { return m.id() == id; });
  if (it == m_meshes.end()) {
    throw std::invalid_argument(
        absl::StrFormat("Not IPU device mesh with ID: %i", id));
  }
  return *it;
}

const IpuDeviceMesh& IpuDeviceMeshManager::find(std::vector<IdType> ids) const {
  std::sort(ids.begin(), ids.end());
  auto it = std::find_if(
      m_meshes.begin(), m_meshes.end(),
      [&ids](const IpuDeviceMesh& m) { return m.info().ipuIds() == ids; });
  if (it == m_meshes.end()) {
    const std::string s = absl::StrJoin(ids.begin(), ids.end(), ", ");
    throw std::out_of_range(
        absl::StrFormat("Not IPU device mesh with IPU IDs: [%s]", s));
  }
  return *it;
}

const IpuDeviceMesh& IpuDeviceMeshManager::defaultMesh(
    std::size_t num_ipus) const {
  CHECK_GT(num_ipus, 0);
  auto it = std::find_if(
      m_meshes.begin(), m_meshes.end(),
      [&num_ipus](const IpuDeviceMesh& m) { return m.size() == num_ipus; });
  if (it == m_meshes.end()) {
    throw std::out_of_range(
        absl::StrFormat("Not IPU device mesh found with %i IPUs", num_ipus));
  }
  return *it;
}

std::size_t IpuDeviceMeshManager::fromMeshIdToIndex(IdType mesh_id) const {
  // TODO: cache the mapping.
  for (std::size_t idx = 0; idx < m_meshes.size(); ++idx) {
    if (m_meshes[idx].id() == mesh_id) {
      return idx;
    }
  }
  throw std::out_of_range(
      absl::StrFormat("Not IPU device mesh found with ID: %i", mesh_id));
}

}  // namespace poplarplugin
}  // namespace xla
