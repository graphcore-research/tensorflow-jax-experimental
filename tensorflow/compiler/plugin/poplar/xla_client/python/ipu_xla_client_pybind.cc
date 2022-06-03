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
  py::class_<IpuConfig> ipu_config(m, "IpuConfig");
  ipu_config.def(py::init<>()).def_readwrite("num_ipus", &IpuConfig::num_ipus);

  py::class_<IpuDevice, PjRtDevice, ClientAndPtr<IpuDevice>>(m, "IpuDevice")
      .def("__repr__", [](const IpuDevice& device) {
        return absl::StrFormat("IpuDevice(id=%i)", device.id());
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