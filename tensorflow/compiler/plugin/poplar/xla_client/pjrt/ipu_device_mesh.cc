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
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/core/platform/default/logging.h"

namespace xla {
namespace poplarplugin {
using IdType = IpuDeviceMeshInfo::IdType;

IpuDeviceMeshInfo::IpuDeviceMeshInfo(
    IdType id, const std::vector<IdType>& ipu_ids, const poplar::Target& target,
    const std::optional<poplar::IPUModel>& ipu_model_desc)
    : m_device_index{0},
      m_mesh_id{id},
      m_ipu_ids{ipu_ids},
      m_target{target},
      m_ipu_model_desc{ipu_model_desc} {
  // Single IPU case.
  if (m_ipu_ids.size() == 0) {
    m_ipu_ids.push_back(m_mesh_id);
  }
  if (m_ipu_ids.size() == 1) {
    CHECK_EQ(m_mesh_id, m_ipu_ids[0]);
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

bool IpuDeviceMeshInfo::isIn(IdType id) const noexcept {
  return std::find(m_ipu_ids.begin(), m_ipu_ids.end(), id) != m_ipu_ids.end();
}

bool IpuDeviceMeshInfo::overlaps(
    const IpuDeviceMeshInfo& mesh_info) const noexcept {
  return this->overlaps(mesh_info.ipu_ids());
}
bool IpuDeviceMeshInfo::overlaps(
    const std::vector<IdType>& ipu_ids) const noexcept {
  for (const auto id : ipu_ids) {
    // Common IPU between the two meshes?
    if (this->isIn(id)) {
      return true;
    }
  }
  // None found!
  return false;
}

IpuDeviceMesh::IpuDeviceMesh(
    poplar::Device device, const std::vector<IdType>& child_ipu_ids,
    const std::optional<poplar::IPUModel>& ipu_model_desc)
    : m_mesh_info{device.getId(), child_ipu_ids, device.getTarget(),
                  ipu_model_desc},
      m_device{std::move(device)} {}

bool IpuDeviceMesh::IsAttached() const noexcept {
  return m_device.isAttached();
}

IpuDeviceMeshManager::IpuDeviceMeshManager(std::vector<IpuDeviceMesh> meshes)
    : m_meshes{std::move(meshes)} {
  // TODO: sort meshes?
  // Create mesh id to index reverse map, and set device index.
  for (std::size_t idx = 0; idx < m_meshes.size(); ++idx) {
    auto& m = m_meshes[idx];
    m_mesh_id_to_index_map[m.id()] = idx;
    m.info().set_local_device_index(idx);
  }
  // Cache of overlapping meshes.
  for (const auto& m0 : m_meshes) {
    std::vector<IdType> overlap_ids;
    // Find overlapping meshes.
    for (const auto& m1 : m_meshes) {
      if (m0.id() != m1.id() && m0.overlaps(m1)) {
        overlap_ids.push_back(m1.id());
      }
    }
    m_mesh_overlap_map[m0.id()] = std::move(overlap_ids);
  }
}

IpuDeviceMeshManager::IpuDeviceMeshManager(
    IpuDeviceMeshManager&& rhs) noexcept {
  // Make sure there is no on-going device management.
  std::scoped_lock l(rhs.m_device_mutex);
  m_meshes = std::move(rhs.m_meshes);
  m_mesh_id_to_index_map = std::move(rhs.m_mesh_id_to_index_map);
  m_mesh_overlap_map = std::move(rhs.m_mesh_overlap_map);
}

void IpuDeviceMeshManager::clear() {
  this->DetachAll();
  m_meshes.clear();
  m_mesh_id_to_index_map.clear();
  m_mesh_overlap_map.clear();
}

StatusOr<IpuDeviceMeshManager> IpuDeviceMeshManager::CreateCpuManager() {
  // not yet supported.
  throw std::runtime_error("IPU device mesh `CPU` devices not yet supported");
  return IpuDeviceMeshManager();
}

StatusOr<IpuDeviceMeshManager> IpuDeviceMeshManager::CreateIpuModelManager(
    int num_devices, int num_tiles, const std::string& version) {
  // By default, 1 IPU model device.
  if (num_devices < 0) {
    num_devices = 1;
  }
  if (num_devices <= 0 || num_devices > 2) {
    return InvalidArgument(
        "IPU model device manager only supporting 1 or 2 IPU devices.");
  }
  // IPU model description to create.
  poplar::IPUModel ipu_model_desc(version.c_str());
  ipu_model_desc.tilesPerIPU = num_tiles;
  // Only support single IPU for now. TODO: multi-ipus.
  std::vector<IpuDeviceMesh> meshes;
  for (int id = 0; id < num_devices; ++id) {
    meshes.push_back(
        IpuDeviceMesh(ipu_model_desc.createDevice(/*optionFlags=*/{},
                                                  /*accurateHalf=*/false,
                                                  /*deviceManagerId=*/id),
                      {}, ipu_model_desc));
  }
  return IpuDeviceMeshManager(std::move(meshes));
}

StatusOr<IpuDeviceMeshManager> IpuDeviceMeshManager::CreateIpuManager(
    int num_devices) {
  if (num_devices == 0) {
    return InvalidArgument("Please provide non-zero number of IPUs to manage.");
  }
  TF_ASSIGN_OR_RETURN(auto all_manager,
                      IpuDeviceMeshManager::CreateIpuManagerWithAll());
  const int num_ipus = all_manager.count(1);
  //  All IPUs requested => simple case, just try attaching them all.
  if (num_devices < 0 || num_devices == num_ipus) {
    // Always try attaching for consistency?
    if (all_manager.AttachAll()) {
      return all_manager;
    }
    return ResourceExhausted(
        "Could not create IPU manager with %u IPUs attached.", num_ipus);
  }
  if (num_ipus < num_devices) {
    return FailedPrecondition(
        "Can not create IPU manager with %u devices: only %u IPUs available.",
        num_devices, num_ipus);
  }
  // All configs currently supported. TODO: global variable.
  const std::set<int> num_devices_supported = {1, 2, 4, 8, 16, 32, 64};
  if (num_devices_supported.find(num_devices) == num_devices_supported.end()) {
    return FailedPrecondition(
        "Can not create IPU manager with %u IPU devices. Only 2^N "
        "configurations supported.",
        num_ipus);
  }
  all_manager.clear();
  // Try finding a collection of IPUs available.
  for (int id = 0; id < num_ipus; id += num_devices) {
    // Slice of IPUs!
    TF_ASSIGN_OR_RETURN(auto manager,
                        IpuDeviceMeshManager::CreateIpuManagerWithAll());
    auto manager_sub = manager.FilterVisibleDevices(id, id + num_devices);
    if (manager_sub.AttachAll()) {
      return manager_sub;
    }
  }
  // Nope!
  return ResourceExhausted(
      "Can not create IPU manager with %u IPUs. No device available.",
      num_devices);
}

StatusOr<IpuDeviceMeshManager> IpuDeviceMeshManager::CreateIpuManager(
    const std::set<int>& visible_devices) {
  if (visible_devices.size() == 0) {
    return InvalidArgument("Provide non-empty set of IPU visible devices.");
  }
  TF_ASSIGN_OR_RETURN(auto all_manager,
                      IpuDeviceMeshManager::CreateIpuManagerWithAll());
  const int num_ipus = all_manager.count(1);
  const std::string visible_ipus_str = absl::StrJoin(visible_devices, ", ");
  // Invalid visible devices mask.
  if (*visible_devices.begin() < 0 || *visible_devices.rbegin() >= num_ipus) {
    return InvalidArgument("Invalid IPU visible devices: {%s}.",
                           visible_ipus_str);
  }
  // Keep only visible devices.
  auto manager_visible = all_manager.FilterVisibleDevices(visible_devices);
  if (manager_visible.AttachAll()) {
    return manager_visible;
  }
  return ResourceExhausted(
      "Can not create IPU manager with IPUs {%s} attached.", visible_ipus_str);
}

StatusOr<IpuDeviceMeshManager> IpuDeviceMeshManager::CreateIpuManagerWithAll() {
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

IpuDeviceMeshManager IpuDeviceMeshManager::FilterVisibleDevices(
    const std::set<int>& visible_devices) {
  // Sorted vector of visible ids.
  const auto visible_ids =
      std::vector<IdType>(visible_devices.begin(), visible_devices.end());
  std::vector<IpuDeviceMesh> meshes;
  meshes.reserve(m_meshes.size());
  // Move meshes subset of visible devices.
  for (std::size_t idx = 0; idx < m_meshes.size(); ++idx) {
    const auto& ipu_ids = m_meshes[idx].info().ipu_ids();
    const bool is_mesh_subset = std::includes(
        visible_ids.begin(), visible_ids.end(), ipu_ids.begin(), ipu_ids.end());
    if (is_mesh_subset) {
      meshes.push_back(std::move(m_meshes[idx]));
    }
  }
  // Clear the current IPU mesh manager.
  this->clear();
  // New IPU mesh manager with subset of meshes.
  return IpuDeviceMeshManager(std::move(meshes));
}
IpuDeviceMeshManager IpuDeviceMeshManager::FilterVisibleDevices(int start_id,
                                                                int end_id) {
  // Visible devices corresponding to the range [start, end).
  std::set<int> visible_devices;
  for (int id = start_id; id < end_id; ++id) {
    visible_devices.insert(id);
  }
  return this->FilterVisibleDevices(visible_devices);
}

bool IpuDeviceMeshManager::IsIpuHardwareAvailable() noexcept {
  return IpuDeviceMeshManager::NumIpuHardwareAvailable() > 0;
}

std::size_t IpuDeviceMeshManager::NumIpuHardwareAvailable() noexcept {
  const auto poplar_manager = poplar::DeviceManager::createDeviceManager();
  const std::size_t num_devices =
      poplar_manager.getDevices(poplar::TargetType::IPU, 1).size();
  return num_devices;
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
        absl::StrFormat("No IPU device mesh with ID: %i", id));
  }
  return *it;
}

const IpuDeviceMesh& IpuDeviceMeshManager::find(std::vector<IdType> ids) const {
  std::sort(ids.begin(), ids.end());
  auto it = std::find_if(
      m_meshes.begin(), m_meshes.end(),
      [&ids](const IpuDeviceMesh& m) { return m.info().ipu_ids() == ids; });
  if (it == m_meshes.end()) {
    const std::string s = absl::StrJoin(ids.begin(), ids.end(), ", ");
    throw std::out_of_range(
        absl::StrFormat("No IPU device mesh with IPU IDs: [%s]", s));
  }
  return *it;
}

const IpuDeviceMesh& IpuDeviceMeshManager::find(
    const DeviceAssignment& device_assignment) const {
  // Convert device indexes to local IPU hardware ids.
  std::vector<IdType> local_ipu_ids;
  local_ipu_ids.reserve(device_assignment.num_elements());
  for (const auto device_index : device_assignment) {
    local_ipu_ids.push_back(m_meshes.at(device_index).local_hardware_id());
  }
  return this->find(std::move(local_ipu_ids));
}

std::size_t IpuDeviceMeshManager::count(std::size_t mesh_size) const noexcept {
  return std::count_if(
      m_meshes.begin(), m_meshes.end(),
      [mesh_size](const auto& m) { return m.size() == mesh_size; });
}

const IpuDeviceMesh& IpuDeviceMeshManager::default_mesh(
    std::size_t num_ipus) const {
  CHECK_GT(num_ipus, 0);
  auto it = std::find_if(
      m_meshes.begin(), m_meshes.end(),
      [&num_ipus](const IpuDeviceMesh& m) { return m.size() == num_ipus; });
  if (it == m_meshes.end()) {
    throw std::out_of_range(
        absl::StrFormat("No IPU device mesh found with %i IPUs", num_ipus));
  }
  return *it;
}

std::size_t IpuDeviceMeshManager::FromMeshIdToIndex(IdType mesh_id) const {
  auto it = m_mesh_id_to_index_map.find(mesh_id);
  if (it == m_mesh_id_to_index_map.end()) {
    throw std::out_of_range(
        absl::StrFormat("No IPU device mesh found with ID: %i", mesh_id));
  }
  return it->second;
}

const std::vector<IdType>& IpuDeviceMeshManager::OverlappingMeshIds(
    IdType mesh_id) const {
  return m_mesh_overlap_map.at(mesh_id);
}

bool IpuDeviceMeshManager::Attach(IdType mesh_id,
                                  bool force_detach_overlapping) const {
  std::scoped_lock l(m_device_mutex);
  // Already attached => by-pass all checks.
  if (this->find(mesh_id).IsAttached()) {
    return true;
  }
  // Tracepoint only when not-already attached.
  TENSORFLOW_TRACEPOINT();
  // Detach all overlapping IPU meshes.
  if (force_detach_overlapping) {
    const auto& overlapping_mesh_ids = this->OverlappingMeshIds(mesh_id);
    // TODO: benchmark performance of detaching? Use boolean cache?
    for (const auto overlap_id : overlapping_mesh_ids) {
      this->find(overlap_id).device().detach();
    }
  }
  // Try finally attaching the device!
  return this->find(mesh_id).device().attach();
}
bool IpuDeviceMeshManager::IsAttached(IdType mesh_id) const {
  std::scoped_lock l(m_device_mutex);
  return this->find(mesh_id).IsAttached();
}
bool IpuDeviceMeshManager::AttachAll() const {
  std::scoped_lock l(m_device_mutex);
  TENSORFLOW_TRACEPOINT();
  const std::size_t num_ipus = this->count(1);
  // Check if single IPUs are already attached?
  bool already_attached = true;
  for (std::size_t idx = 0; idx < num_ipus; ++idx) {
    already_attached &= m_meshes[idx].IsAttached();
  }
  if (already_attached) {
    return true;
  }
  // Not the case: detach all, start from scratch!
  for (const auto& m : m_meshes) {
    m.device().detach();
  }
  // Try attaching all single IPUs...
  for (std::size_t idx = 0; idx < num_ipus; ++idx) {
    if (!m_meshes[idx].device().attach()) {
      return false;
    }
  }
  return true;
}

void IpuDeviceMeshManager::Detach(IdType mesh_id) const {
  std::scoped_lock l(m_device_mutex);
  TENSORFLOW_TRACEPOINT();
  const auto& mesh = this->find(mesh_id);
  // NOTE: bug in Poplar if calling detach on std::move device.
  if (mesh.device().isAttached()) {
    mesh.device().detach();
  }
}
void IpuDeviceMeshManager::DetachAll() const {
  std::scoped_lock l(m_device_mutex);
  TENSORFLOW_TRACEPOINT();
  for (const auto& m : m_meshes) {
    // NOTE: bug in Poplar if calling detach on std::move device.
    if (m.device().isAttached()) {
      m.device().detach();
    }
  }
}

}  // namespace poplarplugin
}  // namespace xla
