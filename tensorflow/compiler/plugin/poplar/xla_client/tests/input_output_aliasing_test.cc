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

#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/passes/input_output_aliasing.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using InputOutputAliasingTest = HloTestBase;

TEST_F(InputOutputAliasingTest, BasicTest) {
  std::string hlo = R"(
HloModule m

ENTRY entry {
  p0 = f16[1]{0} parameter(0)
  p1 = f16[2]{0} parameter(1)
  p2 = f16[3]{0} parameter(2)
  ROOT t = (f16[1], f16[3]) tuple(p0, p2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo));
  
  auto& io_config = module->input_output_alias_config();

  // Parameter 0 & 2 has alias
  io_config.SetUpAlias({0}, 0, {});
  io_config.SetUpAlias({1}, 2, {});

  ASSERT_TRUE(InputOutputAliasing().Run(module.get()).ValueOrDie());

  auto config = module->config();

  ASSERT_EQ(config.argument_input_indices().size(), 1);
  ASSERT_EQ(config.argument_input_indices().at(0), 1);

  ASSERT_EQ(config.resource_input_indices().size(), 2);
  ASSERT_EQ(config.resource_input_indices().at(0), 0);
  ASSERT_EQ(config.resource_input_indices().at(1), 2);

  ASSERT_EQ(config.resource_input_initialized().size(), 2);
  ASSERT_EQ(config.resource_input_initialized().at(0), true);
  ASSERT_EQ(config.resource_input_initialized().at(1), true);

  ASSERT_EQ(config.resource_update_to_input_index().size(), 2);
  ASSERT_EQ(config.resource_update_to_input_index().at(0), 0);
  ASSERT_EQ(config.resource_update_to_input_index().at(1), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
