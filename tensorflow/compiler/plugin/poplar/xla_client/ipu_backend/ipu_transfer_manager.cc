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
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_transfer_manager.h"

#include <memory>

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform_id.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace xla {
namespace poplarplugin {

cpu::runtime::XfeedManager* GetXfeedManager(int device_ordinal) {
  static auto* managers =
      new absl::flat_hash_map<int, cpu::runtime::XfeedManager*>();
  static absl::Mutex* mutex = new absl::Mutex();

  absl::MutexLock lock(mutex);
  auto it = managers->find(device_ordinal);
  if (it == managers->end()) {
    it = managers->emplace(device_ordinal, new cpu::runtime::XfeedManager())
             .first;
  }
  return it->second;
}

IpuTransferManager::IpuTransferManager()
    : GenericTransferManager(kIpuPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status IpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();

  if (!shape.IsTuple()) {
    int64_t size = GetByteSizeRequirement(shape);
    return TransferBufferToInfeed(executor, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()));
  }

  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape = ShapeUtil::GetSubshape(shape, {i});
    int64_t tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
    TF_ASSIGN_OR_RETURN(
        cpu::runtime::XfeedBuffer * buffer,
        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                       literal.untyped_data({i})));

    xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});
  }

  return Status::OK();
}

Status IpuTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  const auto& literal_shape = literal.shape();

  if (!literal_shape.IsTuple()) {
    int64_t size = GetByteSizeRequirement(literal_shape);

    TF_ASSIGN_OR_RETURN(
        Shape received_shape,
        TransferArrayBufferFromOutfeed(executor, literal.untyped_data(), size));
    TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
        << "Shape received from outfeed "
        << ShapeUtil::HumanString(received_shape)
        << " did not match the shape that was requested for outfeed: "
        << ShapeUtil::HumanString(literal_shape);
    TF_RET_CHECK(size == GetByteSizeRequirement(received_shape));
    *literal.mutable_shape_do_not_use() = received_shape;
    return Status::OK();
  }

  if (ShapeUtil::IsNestedTuple(literal_shape)) {
    return Unimplemented(
        "Nested tuple outfeeds are not yet implemented on IPU.");
  }

  std::vector<std::pair<void*, int64_t>> buffer_data;
  for (int64_t i = 0; i < literal_shape.tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(literal_shape, i);
    int64_t size = GetByteSizeRequirement(tuple_element_shape);
    buffer_data.push_back({literal.untyped_data({i}), size});
  }

  TF_ASSIGN_OR_RETURN(Shape received_shape,
                      TransferTupleBuffersFromOutfeed(executor, buffer_data));

  TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal_shape))
      << "Shape received from outfeed "
      << ShapeUtil::HumanString(received_shape)
      << " did not match the shape that was requested for outfeed: "
      << ShapeUtil::HumanString(literal_shape);
  TF_RET_CHECK(GetByteSizeRequirement(literal_shape) ==
               GetByteSizeRequirement(received_shape));

  TF_RET_CHECK(ShapeUtil::Equal(literal.shape(), literal_shape));
  return Status::OK();
}

Status IpuTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                                  int64_t size,
                                                  const void* source) {
  TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(executor, size, source));

  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});

  return Status::OK();
}

StatusOr<cpu::runtime::XfeedBuffer*>
IpuTransferManager::TransferBufferToInfeedInternal(se::StreamExecutor* executor,
                                                   int64_t size,
                                                   const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %d bytes", size);
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %d",
                           size);
  }

  int32 size_32 = static_cast<int32>(size);

  auto* queued_buffer = new IpuInfeedBuffer(size_32);
  std::memcpy(queued_buffer->data(), source, size);

  return queued_buffer;
}

StatusOr<Shape> IpuTransferManager::TransferTupleBuffersFromOutfeed(
    se::StreamExecutor* executor,
    absl::Span<const std::pair<void*, int64_t>> buffer_data) {
  return TransferBuffersFromOutfeedInternal(executor, buffer_data,
                                            /*is_tuple=*/true);
}

StatusOr<Shape> IpuTransferManager::TransferArrayBufferFromOutfeed(
    se::StreamExecutor* executor, void* destination, int64_t size_bytes) {
  return TransferBuffersFromOutfeedInternal(
      executor, {{destination, size_bytes}}, /*is_tuple=*/false);
}

StatusOr<Shape> IpuTransferManager::TransferBuffersFromOutfeedInternal(
    se::StreamExecutor* executor,
    absl::Span<const std::pair<void*, int64_t>> buffer_data, bool is_tuple) {
  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());

  std::vector<Shape> outfed_shapes;
  for (auto b : buffer_data) {
    auto* user_buffer = b.first;
    auto byte_size = b.second;
    auto outfed_buffer = reinterpret_cast<IpuOutfeedBuffer*>(
        xfeed_manager->outfeed()->BlockingDequeueBuffer());
    TF_RET_CHECK(outfed_buffer->length() == byte_size);

    TF_ASSIGN_OR_RETURN(Shape outfed_shape, outfed_buffer->shape());
    std::memcpy(user_buffer, outfed_buffer->data(), byte_size);
    xfeed_manager->outfeed()->ReleaseCurrentBuffer(
        byte_size, outfed_buffer->data(), outfed_shape);

    outfed_shapes.emplace_back(outfed_shape);
  }
  if (is_tuple) {
    return ShapeUtil::MakeTupleShape(outfed_shapes);
  }
  TF_RET_CHECK(outfed_shapes.size() == 1);
  return std::move(outfed_shapes[0]);
}

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateIpuTransferManager() {
  return absl::make_unique<xla::poplarplugin::IpuTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      xla::poplarplugin::kIpuPlatformId, &CreateIpuTransferManager);
  return true;
}

static bool module_initialized = InitModule();
