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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_CUSTOM_CALL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_CUSTOM_CALL_H_

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/normalise_image.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/onehot.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla::poplarplugin::algebraic_simplifier::custom_call {

StatusOr<bool> ElideStatefulGradientAccumulate(HloStatefulGradientAccumulate*);
StatusOr<std::unique_ptr<HloInstruction>> FoldPaddingIntoMaxPool(
    HloMaxPoolInstruction*);
StatusOr<bool> FoldPaddingIntoMaxPoolGrad(HloMaxPoolGradInstruction*);
StatusOr<std::unique_ptr<HloInstruction>> FoldCastIntoNormaliseImage(
    HloNormaliseImage*);
}  // namespace xla::poplarplugin::algebraic_simplifier::custom_call

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_CUSTOM_CALL_H_
