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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PRNG_SEED_STATE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PRNG_SEED_STATE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {

// Utility type for managing prng seed changes via the StochasticRoundingMethod
// flag.
class PrngSeedState {
 public:
  // Create a PrngSeedState from single or multiple seeds. This corresponds
  // to whether we're running with replication (2 seeds) or not (1 seed).
  static PrngSeedState SetupSeed(poplar::Graph& graph, poplar::Tensor& seed,
                                 poplar::program::Sequence& seq);
  static PrngSeedState SetupSeeds(poplar::Graph& graph,
                                  poplar::Tensor& identical_seed,
                                  poplar::Tensor& differing_seed,
                                  poplar::program::Sequence& seq);

  PrngSeedState() = default;

  StochasticRoundingMethod GetStochasticRoundingMethod() const;

  // Set the current stochastic rounding method, without performing any seed
  // changes. This can be used to set the SR method explicitly when control
  // flow programs make it difficult to do so by calling
  // ChangeStochasticRoundingMethod.
  void SetStochasticRoundingMethod(const StochasticRoundingMethod& method);

  // Change the StochasticRoundingMethod to the given type, switching to
  // the appropriate seed. Changes are only made if the method differs to
  // the current one and is not StochasticRoundingMethod_Any.
  // Note, we need to be careful to call this function in the same order that
  // the poplar programs are executed, to make sure that we're actually
  // switching between different SR methods. Consecutive switches between the
  // same SR method can cause the seed values to become inconsistent.
  bool ChangeStochasticRoundingMethod(
      const StochasticRoundingMethod& new_method,
      poplar::program::Sequence& seq,
      const poplar::DebugNameAndId& debug_name_and_id = {});

 private:
  PrngSeedState(poplar::Graph& graph,
                const StochasticRoundingMethod& initial_method,
                poplar::Tensor& identical_hw_seed,
                poplar::Tensor& differing_hw_seed);

  std::unique_ptr<poputil::graphfn::TensorFunction> change_hw_seeds_;
  // Using separate functions for on/off to reduce the size of the compute
  // graph.
  std::unique_ptr<poputil::graphfn::VoidFunction> set_sr_off_;
  std::unique_ptr<poputil::graphfn::VoidFunction> set_sr_on_;

  poplar::Tensor identical_hw_seed_;
  poplar::Tensor differing_hw_seed_;

  StochasticRoundingMethod stochastic_rounding_method_ =
      StochasticRoundingMethod_Undefined;
};

// Debug utility for checking whether the state of the seed matches the given
// StochasticRoundingMethod, terminating poplar program execution if not. Note
// that using this comes with a large overhead.
// Returns whether the given sequence is updated.
bool AssertStochasticRoundingMethod(poplar::Graph& graph,
                                    const StochasticRoundingMethod& method,
                                    poplar::program::Sequence& seq,
                                    const std::string& inst_name = "");

// Debug utility for checking whether stochastic rounding is enabled or not.
// Terminates poplar program execution if the runtime state doesn't match
// `enabled`.
void AssertStochasticRoundingEnabled(poplar::Graph& graph, bool enabled,
                                     poplar::program::Sequence& seq,
                                     const std::string& inst_name = "");

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PRNG_SEED_STATE_H_
