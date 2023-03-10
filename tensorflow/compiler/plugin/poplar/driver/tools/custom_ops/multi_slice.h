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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_SLICE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_SLICE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloMultiSliceInstruction : public HloPoplarInstruction {
 public:
  explicit HloMultiSliceInstruction(const Shape& shape,
                                    HloInstruction* const input,
                                    HloInstruction* const indices,
                                    bool indices_are_sorted = false);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  // Run consumers
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  // Whether or not the given indices are sorted.
  bool GetIndicesAreSorted() const { return indices_are_sorted_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const bool indices_are_sorted_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloStaticMultiSliceInstruction : public HloPoplarInstruction {
 public:
  explicit HloStaticMultiSliceInstruction(const Shape& shape,
                                          HloInstruction* const input,
                                          absl::Span<const int64_t> indices);
  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;
  const std::vector<int64_t>& GetIndices() const { return indices_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const std::vector<int64_t> indices_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloMultiUpdateInstruction : public HloPoplarInstruction {
 public:
  explicit HloMultiUpdateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     bool is_update_add = false,
                                     bool indices_are_sorted = false);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

  // Whether or not the given indices are sorted.
  bool GetIndicesAreSorted() const { return indices_are_sorted_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const bool indices_are_sorted_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloMultiUpdateAddInstruction : public HloMultiUpdateInstruction {
 public:
  explicit HloMultiUpdateAddInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool indices_are_sorted);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

class HloStaticMultiUpdateAddInstruction : public HloPoplarInstruction {
 public:
  explicit HloStaticMultiUpdateAddInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const int64_t> indices);
  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;
  const absl::Span<const int64_t> GetIndices() const { return indices_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  const std::vector<int64_t> indices_;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateMultiSlice(
    const Shape& shape, HloInstruction* const input,
    HloInstruction* const indices, bool indices_are_sorted = false);

std::unique_ptr<HloInstruction> CreateStaticMultiSlice(
    const Shape& shape, HloInstruction* const input,
    absl::Span<const int64_t> indices);

std::unique_ptr<HloInstruction> CreateMultiUpdate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool indices_are_sorted = false);

std::unique_ptr<HloInstruction> CreateMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool indices_are_sorted = false);

std::unique_ptr<HloInstruction> CreateStaticMultiUpdateAdd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const int64_t> indices);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_MULTI_SLICE_H_
