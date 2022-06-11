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

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_compiler.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_executor.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_platform.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_transfer_manager.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace poplarplugin {
namespace {

class IpuBackendTest : public ClientLibraryTestBase {
 public:
  IpuBackendTest()
      : ClientLibraryTestBase(PlatformUtil::GetPlatform("IPU").ValueOrDie()) {}
};

TEST_F(IpuBackendTest, TestClassType) {
  auto* platform = dynamic_cast<IpuPlatform*>(client_->backend().platform());
  EXPECT_NE(platform, nullptr);

  auto* compiler = dynamic_cast<IpuCompiler*>(client_->backend().compiler());
  EXPECT_NE(compiler, nullptr);

  auto* transfer_manager =
      dynamic_cast<IpuTransferManager*>(client_->backend().transfer_manager());
  EXPECT_NE(transfer_manager, nullptr);

  EXPECT_GT(client_->backend().device_count(), 0);
  auto* executor = dynamic_cast<IpuExecutor*>(
      client_->backend().stream_executor(0).ValueOrDie()->implementation());
  EXPECT_NE(executor, nullptr);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
