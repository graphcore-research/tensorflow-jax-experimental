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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_EXECUTOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_EXECUTOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = stream_executor;

namespace xla {
namespace poplarplugin {

class IpuExecutor : public PoplarExecutor {
 public:
  IpuExecutor();
  ~IpuExecutor() override;

  // Currently, Poplar doesn't support dynamic allocation.
  // Return ok directly here to adapt to the pjrt client.
  Status AllocateEvent(se::Event* event) override { return Status::OK(); }
  Status DeallocateEvent(se::Event* event) override { return Status::OK(); }
  Status RecordEvent(se::Stream* stream, se::Event* event) override {
    return Status::OK();
  }
  Status WaitForEvent(se::Stream* stream, se::Event* event) override {
    return Status::OK();
  }
  se::Event::Status PollForEventStatus(se::Event* event) override {
    return se::Event::Status::kComplete;
  }

  Status RegisterOutfeeds(const TranslatedOutfeedInfos& outfeed_infos) override;

 private:
  void ConnectInfeedsToStreamCallback(
      const TranslatedInfeedInfos& infeed_infos) override;

  Status SetupInfeedReplication(const TranslatedInfeedInfos& infeed_infos) override;

  void ConnectOutfeedToStreamCallback(
      const TranslatedOutfeedInfos& outfeed_infos) override;

  void LaunchInfeedThreads(const TranslatedInfeedInfos& infeed_infos) override;
  void LaunchOutfeedThreads(const TranslatedOutfeedInfos& outfeed_infos) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_IPU_BACKEND_IPU_EXECUTOR_H_
