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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_FIFO_INSERTER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_FIFO_INSERTER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * Pass which inserts FIFO operations between forward and backward pipeline
 * stages.
 */
class PipelineFIFOInserter : public HloModulePass {
 public:
  static StatusOr<bool> OffloadFifos(const HloInstruction* pipeline_op,
                                     bool remote_memory_supported);

  explicit PipelineFIFOInserter(bool remote_memory_supported);

  absl::string_view name() const override { return "pipeline-fifo-inserter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Insert FIFOs for a particular pipeline.
  StatusOr<bool> InsertInPipeline(HloInstruction* pipeline_op);

  bool remote_memory_supported_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
