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

#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {
namespace poplarplugin {

namespace py = pybind11;

PYBIND11_MODULE(ipu_xla_client_pybind, m) {
  py::enum_<poplar::TargetType>(m, "IpuTargetType")
      .value("CPU", poplar::TargetType::CPU)
      .value("IPU", poplar::TargetType::IPU)
      .value("IPU_MODEL", poplar::TargetType::IPU_MODEL);

  py::class_<IpuConfig> ipu_config(m, "IpuConfig");
  ipu_config.def(py::init<>()).def_readwrite("num_ipus", &IpuConfig::num_ipus);
  ipu_config.def(py::init<>()).def_readwrite("always_rearrange_copies_on_the_host", 
                                             &IpuConfig::always_rearrange_copies_on_the_host);
  ipu_config.def(py::init<>()).def_readwrite("prefetch_data_streams", 
                                             &IpuConfig::prefetch_data_streams);
  ipu_config.def(py::init<>()).def_readwrite("num_io_tiles", 
                                             &IpuConfig::num_io_tiles);
  ipu_config.def(py::init<>()).def_readwrite("place_ops_on_io_tiles", 
                                             &IpuConfig::place_ops_on_io_tiles);
  ipu_config.def(py::init<>()).def_readwrite("io_tile_available_memory_proportion", 
                                             &IpuConfig::io_tile_available_memory_proportion);

  py::class_<IpuDevice, PjRtDevice, ClientAndPtr<IpuDevice>>(m, "IpuDevice")
      .def("__repr__", [](const IpuDevice& device) {
        return absl::StrFormat("IpuDevice(id=%i, tiles=%i)", device.id(), device.numTiles());
      })
      .def_property_readonly("target_type", [](const IpuDevice& device) { 
        return device.targetType();
      })
      .def_property_readonly("num_tiles", [](const IpuDevice& device) { 
        return device.numTiles();
      })
      .def_property_readonly("num_worker_contexts", [](const IpuDevice& device) { 
        return device.numWorkerContexts();
      })
      .def_property_readonly("bytes_per_tile", [](const IpuDevice& device) { 
        return device.bytesPerTile();
      })
      .def_property_readonly("tile_clock_frequency", [](const IpuDevice& device) { 
        return device.tileClockFrequency();
      });

  m.def(
      "get_ipu_client",
      [](bool asynchronous, const IpuConfig& ipu_config) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetIpuClient(asynchronous, ipu_config));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true,
      py::arg("ipu_config") = IpuConfig());
}

}  // namespace poplarplugin
}  // namespace xla