/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/extended_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_program.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

using DriverGraph = ExtendedGraph;
using DriverTensor = ExtendedTensor;
using DriverDataStream = ExtendedDataStream;
using DriverRemoteBuffer = ExtendedRemoteBuffer;

using DriverProgram = ExtendedProgram;
using DriverProgramSequence = ExtendedProgramSequence;
using DriverProgramCopy = ExtendedProgramCopy;
using DriverProgramSync = ExtendedProgramSync;
using DriverProgramRepeat = ExtendedProgramRepeat;
using DriverProgramCall = ExtendedProgramCall;
using DriverProgramWriteUndef = ExtendedProgramWriteUndef;

using DriverFunction = ExtendedFunction;

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_
