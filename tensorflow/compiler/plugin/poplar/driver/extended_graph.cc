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

#include <utility>
#include <vector>

#include <poputil/TileMapping.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/extended_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_program.h"
#include "tensorflow/compiler/plugin/poplar/driver/extended_tensor.h"

namespace xla {
namespace poplarplugin {

ExtendedGraph::ExtendedGraph(const poplar::Target& target,
                             poplar::replication_factor replication_factor)
    : poplar::Graph(target, replication_factor) {}

ExtendedGraph::ExtendedGraph(poplar::Graph&& graph)
    : poplar::Graph(std::move(graph)) {}

ExtendedGraph ExtendedGraph::createVirtualGraph(unsigned lowerTile,
                                                unsigned upperTile) {
  auto graph = poplar::Graph::createVirtualGraph(lowerTile, upperTile);
  return {std::move(graph)};
}

ExtendedGraph ExtendedGraph::createVirtualGraph(
    const std::vector<unsigned>& perIpuTiles) {
  auto graph = poplar::Graph::createVirtualGraph(perIpuTiles);
  return {std::move(graph)};
}

ExtendedGraph ExtendedGraph::getTopLevelGraph() {
  auto graph = poplar::Graph::getTopLevelGraph();
  return {std::move(graph)};
}

ExtendedTensor ExtendedGraph::addReplicationIndexConstant(
    const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addReplicationIndexConstant(debugContext);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::clone(const poplar::Type& type,
                                    const poplar::Tensor& t,
                                    const poplar::DebugContext& debugContext,
                                    poplar::TensorCloneMethod method) {
  auto clone = poplar::Graph::clone(type, t, debugContext, method);
  return {std::move(clone)};
}

ExtendedTensor ExtendedGraph::clone(const poplar::Tensor& t,
                                    const poplar::DebugContext& debugContext,
                                    poplar::TensorCloneMethod method) {
  auto clone = poplar::Graph::clone(t, debugContext, method);
  return {std::move(clone)};
}

ExtendedTensor ExtendedGraph::addVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addVariable(type, shape, debugContext);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::addVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    poplar::VariableMappingMethod mappingMethod,
    const poplar::DebugContext& debugContext) {
  auto tensor =
      poplar::Graph::addVariable(type, shape, mappingMethod, debugContext);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::addLinearlyMappedVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addVariable(type, shape, debugContext);
  poputil::mapTensorLinearly(*this, tensor);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::addLinearlyMappedVariable(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    unsigned minElementsPerTile, unsigned grainSize,
    const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addVariable(type, shape, debugContext);
  poputil::mapTensorLinearly(*this, tensor, minElementsPerTile, grainSize);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::addConstantHalf(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape, uint16_t val,
    const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addConstantHalf(type, shape, val, debugContext);
  return {std::move(tensor)};
}

ExtendedTensor ExtendedGraph::addConstantHalf(
    const poplar::Type& type, poplar::ArrayRef<std::size_t> shape,
    const uint16_t* val, const poplar::DebugContext& debugContext) {
  auto tensor = poplar::Graph::addConstantHalf(type, shape, val, debugContext);
  return {std::move(tensor)};
}

void ExtendedGraph::setTileMapping(poplar::VertexRef v, unsigned tileNum) {
  poplar::Graph::setTileMapping(v, tileNum);
}

void ExtendedGraph::setTileMapping(const ExtendedTensor& t, unsigned tileNum) {
  poplar::Graph::setTileMapping(t, tileNum);
}

void ExtendedGraph::setTileMapping(const poplar::Tensor& t, unsigned tileNum) {
  poplar::Graph::setTileMapping(t, tileNum);
}

void ExtendedGraph::setTileMapping(
    const ExtendedTensor& t,
    const poplar::Graph::TileToTensorMapping& mapping) {
  poplar::Graph::setTileMapping(t, mapping);
}

poplar::Graph::TileToTensorMapping ExtendedGraph::getTileMapping(
    ExtendedTensor t) const {
  return poplar::Graph::getTileMapping(t);
}

poplar::Graph::TileToTensorMapping ExtendedGraph::getTileMapping(
    poplar::Tensor t) const {
  return poplar::Graph::getTileMapping(t);
}

poplar::HostFunction ExtendedGraph::addHostFunction(
    poplar::StringRef handle,
    poplar::ArrayRef<poplar::Graph::HostFunctionArgument> inputs,
    poplar::ArrayRef<poplar::Graph::HostFunctionArgument> outputs) {
  return poplar::Graph::addHostFunction(handle, inputs, outputs);
}

ExtendedDataStream ExtendedGraph::addHostToDeviceFIFO(
    poplar::StringRef handle, const poplar::Type& elementType,
    std::size_t numElements, poplar::ReplicatedStreamMode replicatedMode,
    const poplar::OptionFlags& options) {
  auto data_stream = poplar::Graph::addHostToDeviceFIFO(
      handle, elementType, numElements, replicatedMode, options);
  return {std::move(data_stream)};
}

ExtendedDataStream ExtendedGraph::addDeviceToHostFIFO(
    poplar::StringRef handle, const poplar::Type& elementType,
    std::size_t numElements, const poplar::OptionFlags& options) {
  auto data_stream = poplar::Graph::addDeviceToHostFIFO(handle, elementType,
                                                        numElements, options);
  return {std::move(data_stream)};
}

void ExtendedGraph::createHostWrite(poplar::StringRef handle,
                                    const ExtendedTensor& t,
                                    bool rearrangeOnHost) {
  poplar::Graph::createHostWrite(handle, t, rearrangeOnHost);
}

void ExtendedGraph::createHostRead(poplar::StringRef handle,
                                   const ExtendedTensor& t,
                                   bool rearrangeOnHost) {
  poplar::Graph::createHostRead(handle, t, rearrangeOnHost);
}

void ExtendedGraph::setInitialValueHalf(const poplar::Tensor& t,
                                        poplar::ArrayRef<uint16_t> values) {
  poplar::Graph::setInitialValueHalf(t, values);
}

bool ExtendedGraph::addCodelets(poplar::StringRef src,
                                poplar::CodeletFileType type,
                                poplar::StringRef compileFlags,
                                poplar::StringRef targetName) {
  return poplar::Graph::addCodelets(src, type, compileFlags, targetName);
}

void ExtendedGraph::addCodelets(std::stringstream& stream,
                                poplar::StringRef compileFlags,
                                poplar::CodeletFileType type,
                                poplar::StringRef targetName) {
  poplar::Graph::addCodelets(stream, compileFlags, type, targetName);
}

void ExtendedGraph::addCodelets(std::stringstream& stream,
                                poplar::StringRef compileFlags,
                                std::ostream& compileOutput,
                                poplar::CodeletFileType type,
                                poplar::StringRef targetName) {
  poplar::Graph::addCodelets(stream, compileFlags, compileOutput, type,
                             targetName);
}

poplar::VertexRef ExtendedGraph::addVertex(poplar::ComputeSet cs,
                                           poplar::StringRef vertexType) {
  return poplar::Graph::addVertex(cs, vertexType);
}

poplar::VertexRef ExtendedGraph::addVertex(
    poplar::ComputeSet cs, poplar::StringRef vertexType,
    poplar::ArrayRef<poplar::Graph::ConnectionDesc> connections) {
  return poplar::Graph::addVertex(cs, vertexType, connections);
}

void ExtendedGraph::connect(poplar::FieldRef field,
                            const poplar::Tensor& tensor) {
  poplar::Graph::connect(field, tensor);
}

std::vector<std::vector<poplar::Interval>>
ExtendedGraph::getSortedContiguousRegions(
    const poplar::Tensor& t, poplar::ArrayRef<poplar::Interval> regions,
    bool removeAliasedIntervals, std::vector<std::size_t>* aliases) const {
  return poplar::Graph::getSortedContiguousRegions(
      t, regions, removeAliasedIntervals, aliases);
}

void ExtendedGraph::setPerfEstimate(const poplar::VertexRef& v,
                                    std::uint64_t cycles, std::uint64_t flops) {
  poplar::Graph::setPerfEstimate(v, cycles, flops);
}

}  // namespace poplarplugin
}  // namespace xla
