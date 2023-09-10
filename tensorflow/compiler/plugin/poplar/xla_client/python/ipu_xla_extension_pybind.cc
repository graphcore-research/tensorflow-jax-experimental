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

#include <cstddef>
#include <cstdint>

#define VISIBLE_SYMBOL __attribute__((visibility("default")))

VISIBLE_SYMBOL int foo(int x) { return x * x; }

namespace py = pybind11;

/**
 * @brief IPU XLA pybind extension. Empty at the moment!
 */
PYBIND11_MODULE(ipu_xla_extension_pybind, m) { m.def("foo", foo); }
