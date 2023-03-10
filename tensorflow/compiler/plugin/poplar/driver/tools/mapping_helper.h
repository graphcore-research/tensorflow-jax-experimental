/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_

#include <vector>

#include <poplar/Interval.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/core/platform/default/integral_types.h"

using tensorflow::uint32;
using tensorflow::uint64;

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
namespace poplarplugin {

using LinearMapperState = absl::flat_hash_map<DriverGraph*, uint64>;
// A helper class for mapping tensors to the IPU which takes previous
// allocations into account.
class MappingHelper {
 public:
  // Gets distance between first and last tile.
  static uint64 GetMappingWidth(
      const std::vector<std::vector<poplar::Interval>>& mapping);
  // Resize mapping to tile number and circularly rotates it.
  static void RotateMapping(DriverGraph& graph,
                            std::vector<std::vector<poplar::Interval>>& mapping,
                            uint64 offset);
  // Maps the tensor linearly, however the starting tile is dependent on
  // previous allocations.
  static void MapTensorLinearly(LinearMapperState& state, DriverGraph& graph,
                                DriverTensor& tensor);
  static void MapTensorLinearly(LinearMapperState& state, DriverGraph& graph,
                                DriverTensor& tensor,
                                uint32 min_elements_per_tile,
                                uint32 grain_size);
  // Remaps existing tensor mapping so its starting tile is dependent on
  // previous allocations.
  static void RemapTensor(LinearMapperState& state, DriverGraph& graph,
                          DriverTensor& tensor);
  // Return the next tile to be mapped to. When allocating the next tensor, the
  // mapping helper starts allocating after the tile returned.
  // Useful for e.g. spreading vertex mapping based on previous allocations.
  static const uint64 YieldNextTile(LinearMapperState& state,
                                    DriverGraph& graph);

 private:
  static void MapTensorLinearlyImpl(
      LinearMapperState& state, DriverGraph& graph, poplar::Tensor& tensor,
      std::vector<std::vector<poplar::Interval>>& mapping);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MAPPING_HELPER_H_
