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
#include <poplar/IPUModel.hpp>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/default/logging.h"

namespace xla {
namespace poplarplugin {
using IdType = IpuDeviceMeshInfo::IdType;

IpuDeviceMeshInfo::IpuDeviceMeshInfo(IdType id,
                                     const std::vector<IdType>& ipu_ids,
                                     const poplar::Target& target)
    : m_id{id}, m_ipu_ids{ipu_ids}, m_target{target} {
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

IpuDeviceMesh::IpuDeviceMesh(poplar::Device device,
                             const std::vector<IdType>& child_ipu_ids)
    : m_mesh_info{device.getId(), child_ipu_ids, device.getTarget()},
      m_device{std::move(device)} {}

IpuDeviceMeshManager::IpuDeviceMeshManager(std::vector<IpuDeviceMesh> meshes)
    : m_meshes{std::move(meshes)} {}

IpuDeviceMeshManager IpuDeviceMeshManager::createCpuManager() {
  // not yet supported.
  return IpuDeviceMeshManager();
}

IpuDeviceMeshManager IpuDeviceMeshManager::createIpuModelManager() {
  // IPU model description to create.
  poplar::IPUModel ipu_model_desc;
  // Only support single IPU for now. TODO: multi-ipus.
  std::vector<IpuDeviceMesh> meshes;
  meshes.push_back(
      IpuDeviceMesh(ipu_model_desc.createDevice(/*optionFlags=*/{},
                                                /*accurateHalf=*/false,
                                                /*deviceManagerId=*/0),
                    {}));
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

const IpuDeviceMesh& IpuDeviceMeshManager::mesh(IdType id) const {
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

}  // namespace poplarplugin
}  // namespace xla
