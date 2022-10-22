/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_

#include <ostream>

#include "absl/container/flat_hash_map.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/core/lib/core/status.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

enum class ValueReplicaCategory { Unknown = 0, Identical, Differing };
std::ostream& operator<<(std::ostream& stream,
                         const ValueReplicaCategory& category);

using ValueCategoryTree = ShapeTree<ValueReplicaCategory>;
std::ostream& operator<<(std::ostream& stream,
                         const ValueCategoryTree& category_tree);

// Visitor for traversing a module to determine which values of each
// instruction will be identical across replicas.
// Expects calls to be flattened.
class ValuesIdenticalAcrossReplicasVisitor
    : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ValuesIdenticalAcrossReplicasVisitor(
      const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
          category_overrides = {});

  // Return the category mapping generated by the visitor.
  const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
  ValueCategoryMapping() const;

  // Return whether the given computation has already been
  // visited.
  bool Visited(const HloComputation* comp) const;

  Status Postprocess(const HloInstruction* inst) override;

  Status DefaultAction(const HloInstruction* inst) override;

  Status HandleCall(const HloInstruction* inst) override;
  Status HandleConditional(const HloInstruction* inst) override;
  Status HandleCustomCall(const HloInstruction* inst) override;
  Status HandleAllReduce(const HloInstruction* inst) override;
  Status HandleFusion(const HloInstruction* inst) override;
  Status HandleGetTupleElement(const HloInstruction* inst) override;
  Status HandleTuple(const HloInstruction* inst) override;
  Status HandleWhile(const HloInstruction* inst) override;

#define HandleAsReplicaIdentical(TYPE)                       \
  Status Handle##TYPE(const HloInstruction* inst) override { \
    return SetAllInstructionValuesToIdentical(inst);         \
  }

  HandleAsReplicaIdentical(Parameter);
  HandleAsReplicaIdentical(Constant);

#undef HandleAsReplicaIdentical

#define HandleAsReplicaDiffering(TYPE)                       \
  Status Handle##TYPE(const HloInstruction* inst) override { \
    return SetAllInstructionValuesToDiffering(inst);         \
  }

  HandleAsReplicaDiffering(Infeed);
  HandleAsReplicaDiffering(ReplicaId);
  HandleAsReplicaDiffering(Rng);

#undef HandleAsReplicaDiffering

 private:
  Status HandleAllGather(const HloInstruction* inst);
  Status HandleRepeatLoop(const HloInstruction* call,
                          const HloComputation* body, int64_t repeat_count);
  Status HandleUserOp(const HloInstruction* inst);

  // Visit a HloComputation using a specific value category for each of its
  // parameters, returning the value categories of the root instruction and
  // storing the results.
  StatusOr<ValueCategoryTree> VisitSubComputation(const HloComputation* comp,
                                                  const HloInstruction* call);
  StatusOr<ValueCategoryTree> VisitSubComputation(
      const HloComputation* comp,
      const ValueCategoryTree& parameter_categories);
  StatusOr<ValueCategoryTree> VisitSubComputation(
      const HloComputation* comp,
      const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
          parameter_overrides);

  // Create a map of overrides for the parameters of the given HloComputation
  // using the operands of the given HloInstruction. These can be used to find
  // the value categories of `comp` when called by `call`.
  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
  CreateParameterOverridesForCall(const HloInstruction* call,
                                  const HloComputation* comp) const;

  bool AllOperandsIdentical(const HloInstruction* inst) const;

  Status SetAllInstructionValuesToIdentical(const HloInstruction* inst);
  Status SetAllInstructionValuesToDiffering(const HloComputation* computation);
  Status SetAllInstructionValuesToDiffering(const HloInstruction* inst);
  Status SetAllInstructionValuesToIdenticalOrDiffering(
      const HloInstruction* inst, bool identical);

  void SetInstrucionValueToIdenticalOrDiffering(const HloInstruction* inst,
                                                const ShapeIndex& value_index,
                                                bool identical);

  void MarkOverridesAsVisited(
      const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
          category_overrides);

  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      value_category_mapping_;
};

// Analyse the given HloModule to find values which are identical
// across replicas.
class ReplicaIdenticalDataflowAnalysis {
 public:
  // Run the analysis. Requires that `module` be flattened.
  Status Run(const HloModule* module);

  // Run the analysis on a single computation.
  Status AnalyseComputation(const HloComputation* comp);

  // Check whether or not the given comp, and as a result its
  // instructions, have been analysed.
  bool Analysed(const HloComputation* comp) const;

  // Return the ValueReplicaCategory for the given instruction/value_index
  // or an error if the instruction has not been analysed.
  StatusOr<ValueReplicaCategory> ValueCategory(
      const HloInstruction* inst,
      const ShapeIndex& value_index = RootShapeIndex()) const;

  StatusOr<bool> IsValueIdenticalAcrossReplicas(
      const HloInstruction* inst,
      const ShapeIndex& value_index = RootShapeIndex()) const;

 private:
  ValuesIdenticalAcrossReplicasVisitor value_category_visitor_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_
