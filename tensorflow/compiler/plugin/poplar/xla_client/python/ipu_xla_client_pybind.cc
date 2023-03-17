/* Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device_mesh.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {
namespace poplarplugin {

namespace py = pybind11;

PYBIND11_MODULE(ipu_xla_client_pybind, m) {
  // Poplar classes Python bindings.
  py::enum_<poplar::TargetType>(m, "IpuPoplarTargetType")
      .value("CPU", poplar::TargetType::CPU)
      .value("IPU", poplar::TargetType::IPU)
      .value("IPU_MODEL", poplar::TargetType::IPU_MODEL);

  py::class_<poplar::Target>(m, "IpuPoplarTarget")
      .def_property_readonly("type", &poplar::Target::getTargetType)
      .def_property_readonly("system_info",
                             &poplar::Target::getTargetSystemString)
      .def_property_readonly("arch_info",
                             [](const poplar::Target& t) {
                               return std::string(t.getTargetArchString());
                             })
      .def_property_readonly("num_ipus", &poplar::Target::getNumIPUs)
      .def_property_readonly("num_tiles_per_ipu",
                             &poplar::Target::getTilesPerIPU)
      .def_property_readonly("num_worker_contexts",
                             &poplar::Target::getNumWorkerContexts)
      .def_property_readonly("bytes_per_tile", &poplar::Target::getBytesPerTile)
      .def_property_readonly("tile_clock_frequency",
                             &poplar::Target::getTileClockFrequency);

  // IPU device mesh classes.
  using IdType = IpuDeviceMeshInfo::IdType;
  py::class_<IpuDeviceMeshInfo>(m, "IpuDeviceMeshInfo")
      .def(py::init<IdType, const std::vector<IdType>, const poplar::Target&>(),
           py::arg("id"), py::arg("ipu_ids"), py::arg("target"))
      .def("__len__", &IpuDeviceMeshInfo::size)
      .def("__contains__", &IpuDeviceMeshInfo::isIn)
      .def_property_readonly("id", &IpuDeviceMeshInfo::id)
      .def_property_readonly("ipu_ids", &IpuDeviceMeshInfo::ipuIds)
      .def_property_readonly("target", &IpuDeviceMeshInfo::target,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("size", &IpuDeviceMeshInfo::size)
      .def_property_readonly("single", &IpuDeviceMeshInfo::single);

  py::class_<IpuDeviceMesh>(m, "IpuDeviceMesh")
      .def("__len__", &IpuDeviceMesh::size)
      .def("__contains__", &IpuDeviceMesh::isIn)
      .def("overlaps", &IpuDeviceMesh::overlaps, py::arg("mesh"))
      .def_property_readonly("id", &IpuDeviceMesh::id)
      .def_property_readonly("size", &IpuDeviceMesh::size)
      .def_property_readonly("type", &IpuDeviceMesh::type)
      .def_property_readonly("version", &IpuDeviceMesh::version)
      .def_property_readonly("num_tiles_per_ipu",
                             &IpuDeviceMesh::num_tiles_per_ipu)
      .def_property_readonly("info", &IpuDeviceMesh::info,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("target", &IpuDeviceMesh::target,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("is_attached", &IpuDeviceMesh::isAttached);

  py::class_<IpuDeviceMeshManager>(m, "IpuDeviceMeshManager")
      .def("find",
           py::overload_cast<IdType>(&IpuDeviceMeshManager::find, py::const_),
           py::arg("id"), py::return_value_policy::reference_internal)
      .def("find",
           py::overload_cast<std::vector<IdType>>(&IpuDeviceMeshManager::find,
                                                  py::const_),
           py::arg("ids"), py::return_value_policy::reference_internal)
      .def("find",
           py::overload_cast<const DeviceAssignment&>(
               &IpuDeviceMeshManager::find, py::const_),
           py::arg("device_assignment"),
           py::return_value_policy::reference_internal)
      .def("__len__", &IpuDeviceMeshManager::size)
      .def("__getitem__", &IpuDeviceMeshManager::at, py::arg("idx"),
           py::return_value_policy::reference_internal)
      .def("count", &IpuDeviceMeshManager::count, py::arg("mesh_size") = 1)
      .def("default_mesh", &IpuDeviceMeshManager::defaultMesh,
           py::arg("num_ipus"), py::return_value_policy::reference_internal)
      .def("from_mesh_id_to_index", &IpuDeviceMeshManager::fromMeshIdToIndex,
           py::arg("mesh_id"))
      .def("overlapping_mesh_ids", &IpuDeviceMeshManager::overlappingMeshIds,
           py::arg("mesh_id"))
      .def("attach", &IpuDeviceMeshManager::attach, py::arg("mesh_id"),
           py::arg("force_detach_overlapping") = true)
      .def("is_attached", &IpuDeviceMeshManager::isAttached, py::arg("mesh_id"))
      .def("detach", &IpuDeviceMeshManager::detach, py::arg("mesh_id"))
      .def("detach_all", &IpuDeviceMeshManager::detachAll)
      .def_property_readonly("size", &IpuDeviceMeshManager::size)
      .def_property_readonly("type", &IpuDeviceMeshManager::type)
      .def_property_readonly("meshes", &IpuDeviceMeshManager::meshes)
      .def_static("has_local_ipu_hardware",
                  &IpuDeviceMeshManager::hasLocalIpuHardware)
      .def_static("create_ipu_model_manager",
                  &IpuDeviceMeshManager::createIpuModelManager,
                  py::arg("num_tiles") = 4, py::arg("version") = "ipu2")
      .def_static("create_ipu_manager",
                  &IpuDeviceMeshManager::createIpuManager);

  // IPU PjRt client bindings.
  py::class_<IpuPjRtOptions>(m, "IpuPjRtOptions")
      .def(py::init<>())
      .def(py::init<std::optional<std::set<int>>, bool, int, std::string>(),
           py::arg("visible_devices") = std::nullopt,
           py::arg("use_ipu_model") = false, py::arg("ipu_model_num_tiles") = 4,
           py::arg("ipu_model_version") = "ipu2")
      .def_readwrite("visible_devices", &IpuPjRtOptions::visible_devices)
      .def_readwrite("use_ipu_model", &IpuPjRtOptions::use_ipu_model)
      .def_readwrite("ipu_model_num_tiles",
                     &IpuPjRtOptions::ipu_model_num_tiles)
      .def_readwrite("always_rearrange_copies_on_the_host",
                     &IpuPjRtOptions::always_rearrange_copies_on_the_host)
      .def_readwrite("prefetch_data_streams",
                     &IpuPjRtOptions::prefetch_data_streams);

  py::class_<IpuPjRtExecutableRunInfo>(m, "IpuPjRtExecutableRunInfo")
      .def(py::init<>())
      .def(py::init<int, int, int>(), py::arg("mesh_id"),
           py::arg("executable_id") = 0, py::arg("run_id") = 0)
      .def_readwrite("mesh_id", &IpuPjRtExecutableRunInfo::mesh_id)
      .def_readwrite("executable_id", &IpuPjRtExecutableRunInfo::executable_id)
      .def_readwrite("run_id", &IpuPjRtExecutableRunInfo::run_id);

  py::class_<IpuPjRtMeshState>(m, "IpuPjRtMeshState")
      .def(py::init<>())
      .def(py::init<int, int, int>(), py::arg("mesh_id"),
           py::arg("executable_id") = 0, py::arg("run_id") = 0)
      .def_readwrite("mesh_id", &IpuPjRtMeshState::mesh_id)
      .def_readwrite("executable_id", &IpuPjRtMeshState::executable_id)
      .def_readwrite("run_id", &IpuPjRtMeshState::run_id);

  py::class_<IpuPjRtClientState>(m, "IpuPjRtClientState")
      .def(py::init<>())
      .def("__len__", &IpuPjRtClientState::size)
      .def_static("initialize", &IpuPjRtClientState::Initialize,
                  py::arg("mesh_manager"))
      .def("update", &IpuPjRtClientState::Update, py::arg("run_info"),
           py::arg("mesh_manager"))
      .def("is_active_mesh", &IpuPjRtClientState::IsActiveMesh,
           py::arg("mesh_id"))
      .def_property_readonly("active_meshes",
                             &IpuPjRtClientState::active_meshes);

  py::class_<IpuPjRtDevice, PjRtDevice, ClientAndPtr<IpuPjRtDevice>>(
      m, "IpuPjRtDevice")
      .def("__repr__", &IpuPjRtDevice::ToString)
      .def_property_readonly("ipu_mesh_manager",
                             &IpuPjRtDevice::ipu_mesh_manager,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "type", [](const IpuPjRtDevice& d) { return d.device_info().type(); })
      .def_property_readonly(  // backward compatible with legacy client.
          "target_type",
          [](const IpuPjRtDevice& d) { return d.device_info().type(); })
      .def_property_readonly(
          "version",
          [](const IpuPjRtDevice& d) { return d.device_info().version(); })
      .def_property_readonly("num_tiles",
                             [](const IpuPjRtDevice& d) {
                               return d.device_info().target().getTilesPerIPU();
                             })
      .def_property_readonly(
          "num_worker_contexts",
          [](const IpuPjRtDevice& d) {
            return d.device_info().target().getNumWorkerContexts();
          })
      .def_property_readonly(
          "bytes_per_tile",
          [](const IpuPjRtDevice& d) {
            return d.device_info().target().getBytesPerTile();
          })
      .def_property_readonly(
          "tile_clock_frequency", [](const IpuPjRtDevice& d) {
            return d.device_info().target().getTileClockFrequency();
          });

  m.def("create_ipu_device_mesh_manager", CreateIpuDeviceMeshManager,
        py::arg("ipu_options"));
  m.def(
      "get_ipu_client",
      [](bool asynchronous, const IpuPjRtOptions& ipu_options)
          -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetIpuClient(asynchronous, ipu_options));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous"), py::arg("ipu_options"));

  // LEGACY IPU Pjrt bindings.
  // LEGACY IPU Pjrt bindings.
  // LEGACY IPU Pjrt bindings.
  py::class_<IpuConfig> ipu_config(m, "IpuConfig");
  ipu_config.def(py::init<>())
      .def_readwrite("num_ipus", &IpuConfig::num_ipus)
      .def_readwrite("always_rearrange_copies_on_the_host",
                     &IpuConfig::always_rearrange_copies_on_the_host)
      .def_readwrite("prefetch_data_streams", &IpuConfig::prefetch_data_streams)
      .def_readwrite("num_io_tiles", &IpuConfig::num_io_tiles)
      .def_readwrite("place_ops_on_io_tiles", &IpuConfig::place_ops_on_io_tiles)
      .def_readwrite("io_tile_available_memory_proportion",
                     &IpuConfig::io_tile_available_memory_proportion);

  py::class_<IpuDevice, PjRtDevice, ClientAndPtr<IpuDevice>>(m, "IpuDevice")
      .def("__repr__",
           [](const IpuDevice& device) {
             return absl::StrFormat("IpuDevice(id=%i, tiles=%i)", device.id(),
                                    device.numTiles());
           })
      .def_property_readonly(
          "target_type",
          [](const IpuDevice& device) { return device.targetType(); })
      .def_property_readonly(
          "num_tiles",
          [](const IpuDevice& device) { return device.numTiles(); })
      .def_property_readonly(
          "num_worker_contexts",
          [](const IpuDevice& device) { return device.numWorkerContexts(); })
      .def_property_readonly(
          "bytes_per_tile",
          [](const IpuDevice& device) { return device.bytesPerTile(); })
      .def_property_readonly(
          "tile_clock_frequency",
          [](const IpuDevice& device) { return device.tileClockFrequency(); });

  m.def(
      "get_legacy_ipu_client",
      [](bool asynchronous,
         const IpuConfig& ipu_config) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetLegacyIpuClient(asynchronous, ipu_config));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true, py::arg("ipu_config") = IpuConfig());
}

}  // namespace poplarplugin
}  // namespace xla
