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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_TRANSFER_MANAGER_H_

#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace poplarplugin {

cpu::runtime::XfeedManager* GetXfeedManager(int device_ordinal);

class IpuInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit IpuInfeedBuffer(int32 length)
      : length_(length), buffer_(new char[length]) {}

  ~IpuInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override { delete this; }

 private:
  int32 length_;
  char* buffer_;
};

class IpuOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  IpuOutfeedBuffer(int32 length, xla::Shape shape)
      : length_(length),
        status_(std::move(shape)),
        destination_(new char[length]) {}

  IpuOutfeedBuffer(void* destination, int32 length, xla::Shape shape)
      : length_(length), status_(std::move(shape)), destination_(destination) {}

  int32 length() override { return length_; }
  void* data() override { return destination_; }
  void Done(StatusOr<Shape> shape) override {
    delete[] reinterpret_cast<char*>(destination_);
  }

  StatusOr<Shape> shape() const { return status_; }

 private:
  int32 length_;
  StatusOr<Shape> status_;
  void* destination_;
};

class IpuTransferManager : public GenericTransferManager {
 public:
  IpuTransferManager();
  ~IpuTransferManager() override = default;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;

  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal) override;

 private:
  Status TransferBufferToInfeed(se::StreamExecutor* executor, int64_t size,
                                const void* source);

  StatusOr<cpu::runtime::XfeedBuffer*> TransferBufferToInfeedInternal(
      se::StreamExecutor* executor, int64_t size, const void* source);

  // Helper that transfers a tuple of element buffers from the device's outfeed.
  StatusOr<Shape> TransferTupleBuffersFromOutfeed(
      se::StreamExecutor* executor,
      absl::Span<const std::pair<void*, int64_t>> buffer_data);

  // Helper that transfers an array buffer from the device's outfeed.
  StatusOr<Shape> TransferArrayBufferFromOutfeed(se::StreamExecutor* executor,
                                                 void* destination,
                                                 int64_t size_bytes);

  // On success, returns the shape that was transferred from the outfeed -- if
  // is_tuple is true, the returned shape will be a tuple of the returned shapes
  // for the given buffers.
  StatusOr<Shape> TransferBuffersFromOutfeedInternal(
      se::StreamExecutor* executor,
      absl::Span<const std::pair<void*, int64_t>> buffer_data, bool is_tuple);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuTransferManager);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_TRANSFER_MANAGER_H_
