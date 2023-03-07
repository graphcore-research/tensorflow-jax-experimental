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
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_executor.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_transfer_manager.h"

namespace xla {
namespace poplarplugin {

IpuExecutor::IpuExecutor() : PoplarExecutor() {}

IpuExecutor::~IpuExecutor() {}

Status IpuExecutor::RegisterOutfeeds(
    const TranslatedOutfeedInfos& outfeed_infos) {
  return Status::OK();
}

namespace {
class NullPrefetchCallback : public poplar::StreamCallback {
 public:
  explicit NullPrefetchCallback(InfeedAllocator* allocator, uint64 num_bytes)
      : num_bytes_(num_bytes), allocator_(allocator) {
    for (auto& buffer : buffers_) {
      buffer = static_cast<uint8*>(allocator_->AllocateRaw(64, num_bytes));
      std::memset(buffer, 0x2, num_bytes);
    }
  }

  ~NullPrefetchCallback() {
    for (auto& buffer : buffers_) {
      allocator_->DeallocateRaw(buffer);
    }
  }

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    std::memcpy(dest, buffers_[index_], num_bytes_);
    return poplar::StreamCallback::Result::Success;
  }

  void fetch(void* dest) noexcept override {
    // This case shouldn't be hit, if poplar prefetches the data.
    std::memcpy(dest, buffers_[index_], num_bytes_);
  }

  void complete() noexcept override { index_ = (index_ + 1) % 16; }

 private:
  int index_ = 0;
  uint8* buffers_[16];
  const uint64 num_bytes_;
  InfeedAllocator* allocator_;
};

class InfeedPrefetchCallback : public poplar::StreamCallback {
 public:
  InfeedPrefetchCallback(cpu::runtime::XfeedQueueManager* queue,
                         int32 num_bytes)
      : queue_(queue), num_bytes_(num_bytes) {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Try to get a value from the queue.
    auto* buffer = queue_->TryDequeueBuffer();
    if (buffer) {
      CHECK_EQ(buffer->length(), num_bytes_);
      std::memcpy(dst, buffer->data(), buffer->length());
      queue_->ReleaseCurrentBuffer(buffer->length(), buffer->data(), Shape{});
      return poplar::StreamCallback::Result::Success;
    } else {
      return poplar::StreamCallback::Result::NotAvailable;
    }
  }

  void fetch(void* dst) noexcept override {
    auto* buffer = queue_->BlockingDequeueBuffer();
    CHECK_EQ(buffer->length(), num_bytes_);
    std::memcpy(dst, buffer->data(), buffer->length());
    queue_->ReleaseCurrentBuffer(buffer->length(), buffer->data(), Shape{});
  }

  void complete() noexcept override {}

 private:
  cpu::runtime::XfeedQueueManager* queue_;
  const int32 num_bytes_;
};
}  // namespace

// Modified from PoplarExector
// https://github.com/graphcore/tensorflow-jax/blob/52cbc3c2c22fee8176f2e146e72165de38e3c630/tensorflow/compiler/plugin/poplar/driver/poplar_executor.cc#L818
// Key changes:
//  - Use new InfeedPrefetchCallback that interacts with transfer_manager
//  - Remove replication as infeed is not supported through infeed_iterator
//    Maybe will be done via pmap(T66396)
// Relevant infeed/outfeed changes caused by similar reason.
// See
// https://github.com/graphcore/tensorflow-jax/pull/3#pullrequestreview-1049554034
void IpuExecutor::ConnectInfeedsToStreamCallback(
    const TranslatedInfeedInfos& infeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Infeed)) {
    return;
  }

  CHECK_EQ(current_replication_factor_, 1);
  CHECK_EQ(infeed_infos.size(), 1);

  std::unique_lock<std::mutex> l(infeeds_mutex_);
  for (const auto& infeed_info : infeed_infos) {
    const auto& infeed_shape = infeed_info.canonical_info.shape;
    // ((...), token)
    CHECK_EQ(ShapeUtil::IsNestedTuple(infeed_shape), true);
    const auto& shape = ShapeUtil::GetTupleElementShape(infeed_shape, 0);
    // IsNestedTuple will check shape is tuple
    CHECK_EQ(ShapeUtil::IsNestedTuple(shape), false);

    for (auto i = 0; i < shape.tuple_shapes_size(); i++) {
      std::unique_ptr<poplar::StreamCallback> infeed_callback;
      const auto& element_shape = ShapeUtil::GetTupleElementShape(shape, i);
      const auto length = ShapeUtil::ByteSizeOfElements(element_shape);

      if (PoplarXlaFlags::Get().null_data_feed) {
        infeed_callback = absl::make_unique<NullPrefetchCallback>(
            GetInfeedAllocator(), length);
      } else {
        auto* xfeed_manager =
            xla::poplarplugin::GetXfeedManager(device_ordinal());
        infeed_callback = absl::make_unique<InfeedPrefetchCallback>(
            xfeed_manager->infeed(), length);
      }
      current_engine_->connectStreamToCallback(
          GetInfeedCopyHandle(infeed_info.canonical_info.config.feed_id(), i),
          std::move(infeed_callback));
    }
  }
}

Status IpuExecutor::SetupInfeedReplication(
    const TranslatedInfeedInfos& infeed_infos) {
  return Status::OK();
}

void IpuExecutor::ConnectOutfeedToStreamCallback(
    const TranslatedInfeedInfos& outfeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
    return;
  }

  CHECK_EQ(current_replication_factor_, 1);
  CHECK_EQ(outfeed_infos.size(), 1);

  std::unique_lock<std::mutex> l(outfeeds_mutex_);
  for (const auto& outfeed_info : outfeed_infos) {
    const auto& shape = outfeed_info.canonical_info.shape;
    // IsNestedTuple will check shape is tuple
    CHECK_EQ(ShapeUtil::IsNestedTuple(shape), false);

    for (auto i = 0; i < shape.tuple_shapes_size(); i++) {
      current_engine_->connectStreamToCallback(
          GetOutfeedCopyHandle(outfeed_info.canonical_info.config.feed_id(), i),
          [=](void* src) {
            auto* xfeed_manager =
                xla::poplarplugin::GetXfeedManager(device_ordinal());

            const auto& element_shape =
                ShapeUtil::GetTupleElementShape(shape, i);
            const auto length = ShapeUtil::ByteSizeOfElements(element_shape);
            auto* buffer = new IpuOutfeedBuffer(length, element_shape);

            std::memcpy(buffer->data(), src, buffer->length());

            xfeed_manager->outfeed()->EnqueueBuffersAtomically({buffer});
          });
    }
  }
}

void IpuExecutor::LaunchInfeedThreads(
    const TranslatedInfeedInfos& infeed_infos) {
  return;
}

void IpuExecutor::LaunchOutfeedThreads(
    const TranslatedOutfeedInfos& outfeed_infos) {
  return;
}

Status IpuExecutor::AttachDevice() {
  const bool r = ipu_.AttachDevice();
  if (r) {
    return Status::OK();
  }
  return ResourceExhausted("IPU Poplar device could not be attached.");
}
Status IpuExecutor::DetachDevice() {
  ipu_.DetachDevice();
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
