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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/lstm.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/rnn_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Util.hpp>

namespace xla {
namespace poplarplugin {
namespace {
static const size_t basic_lstm_cell_num_units = 4;

poplar::Tensor UnflattenWeight(const poplar::Tensor& t) {
  return t
      .reshape({t.dim(0), basic_lstm_cell_num_units,
                t.dim(1) / basic_lstm_cell_num_units})
      .dimShuffle({1, 0, 2});
}

// The kernel is stored as:
// [input_size + output_size, basic_lstm_cell_num_units * output_size] tensor.
// This extracts the input and output weights.
std::pair<poplar::Tensor, poplar::Tensor> UnpackLstmKernel(
    poplar::Tensor kernel, const size_t input_size, const size_t output_size) {
  poplar::Tensor inputWeights = UnflattenWeight(kernel.slice(0, input_size));
  poplar::Tensor outputWeights =
      UnflattenWeight(kernel.slice(input_size, input_size + output_size));
  return {inputWeights, outputWeights};
}

poplar::Tensor FlattenWeight(const poplar::Tensor& t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// Reverse of UnpackLstmKernel
poplar::Tensor PackLstmKernel(poplar::Tensor input_weights,
                              poplar::Tensor output_weights) {
  return poplar::concat(FlattenWeight(input_weights),
                        FlattenWeight(output_weights));
}

class LstmLayerBaseOp : public PoplarOpDef {
 protected:
  virtual std::vector<poplar::Tensor> GetOutputParams(bool training) {
    return std::vector<poplar::Tensor>{OutputTensorCount()};
  }

  virtual poputil::graphfn::Signature GetSignature(
      const std::vector<poplar::Tensor>& args, bool training) {
    poputil::graphfn::Signature signature;
    int64_t total_param_count = InputTensorCount() + OutputTensorCount();
    const auto& name_list = NameList();

    for (int64_t i = 0; i < total_param_count; ++i) {
      if (i < InputTensorCount()) {
        signature.push_back(poputil::graphfn::input(args[i], name_list[i]));
      } else {
        signature.push_back(poputil::graphfn::created(name_list[i]));
      }
    }
    return signature;
  }

  virtual void SetOutputTensor(DriverGraph& graph,
                               std::vector<poplar::Tensor>& args,
                               const HloInstruction* inst,
                               TensorMap& tensor_map, bool training) {
    for (int64_t j = 0; j < OutputTensorCount(); ++j) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, j,
                                  DriverTensor(args[j + InputTensorCount()])));
    }
  }

 public:
  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    const HloInstruction* inst = tensor_target.tgt;
    const int64_t input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                        GetLstmParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts, GetLstmOpts(inst, res));
    switch (input_index) {
      case 0: {
        // Allocate LSTM input tensor
        return DriverTensor(popnn::lstm::createInput(
            graph, lstm_params, {debug_info}, lstm_opts, &res.planning_cache));
      }
      case 1: {
        // Allocate initial output (h) tensor
        return DriverTensor(popnn::lstm::createInitialOutput(
            graph, lstm_params, {debug_info}, lstm_opts, &res.planning_cache));
      }
      case 2: {
        // Allocate initial cell state (c) tensor
        return DriverTensor(popnn::lstm::createInitialCellState(
            graph, lstm_params, {debug_info}, lstm_opts, &res.planning_cache));
      }
      case 3: {
        // Allocate LSTM weights kernel
        poplar::Tensor input_weights;
        poplar::Tensor output_weights;
        std::tie(input_weights, output_weights) =
            popnn::lstm::createWeightsKernel(graph, lstm_params, {debug_info},
                                             lstm_opts, &res.planning_cache);
        return DriverTensor(PackLstmKernel(input_weights, output_weights));
      }
      case 4: {
        // Allocate LSTM weights biases
        return DriverTensor(popnn::lstm::createWeightsBiases(
            graph, lstm_params, {debug_info}, lstm_opts, &res.planning_cache));
      }
      case 5: {
        // Allocate DynamicLSTM seq_len
        return DriverTensor(popops::createSliceableTensor(
            graph, poplar::INT, {lstm_params.rnn.batchSize}, {0}, {1}, 0,
            name));
      }
      default: {
        return xla::FailedPrecondition(
            "Trying to allocate LstmLayerFwdOp tensor for an index out of "
            "range "
            "%d.",
            input_index);
      }
    }
  }

  virtual StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    DriverProgramSequence seq(debug_info);

    auto lstm_inst = Cast<HloRNNInstruction>(inst);
    bool training = lstm_inst->is_training();

    std::vector<poplar::Tensor> args;
    for (int64_t i = 0; i < InputTensorCount(); ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor input_tensor,
          FindInstructionInput(tensor_map, res, inst, i, seq, debug_info,
                               /*expand_aliasing*/ false));
      args.push_back(input_tensor);
    }

    auto output_args = GetOutputParams(training);
    args.insert(args.end(), output_args.begin(), output_args.end());

    poputil::graphfn::Signature signature = GetSignature(args, training);

    TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                        GetLstmParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts, GetLstmOpts(inst, res));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    poplar::DebugNameAndId debug_name_and_id(debug_info);

    LowerToPoplar(graph, res, inst, lstm_params, lstm_opts, input_size,
                  output_size, training, debug_name_and_id, args, seq);

    SetOutputTensor(graph, args, inst, tensor_map, training);
    return seq;
  }

 protected:
  virtual const std::string ClassName() const = 0;

  virtual std::vector<const char*> NameList() const = 0;

  virtual int64_t InputTensorCount() const = 0;

  virtual int64_t OutputTensorCount() const = 0;

  virtual void LowerToPoplar(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      popnn::lstm::LstmParams lstm_params, const poplar::OptionFlags& lstm_opts,
      const int64_t& input_size, const int64_t& output_size, bool training,
      const poplar::DebugNameAndId& debug_name_and_id,
      std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) = 0;
};

class LstmLayerFwdOp : public LstmLayerBaseOp {
 public:
  LstmLayerFwdOp() = default;

 protected:
  const std::string ClassName() const override { return "LstmLayerFwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {
        "input_seq", "input_h_state", "input_c_state", "kernel",
        "bias",      "output",        "output_c_state"};
    return name_list;
  }

  int64_t InputTensorCount() const override { return 5; }

  int64_t OutputTensorCount() const override { return 2; }

  std::vector<poplar::Tensor> GetOutputParams(bool training) override {
    auto args = LstmLayerBaseOp::GetOutputParams(training);
    if (training) {
      poplar::Tensor intermediates;
      args.push_back(intermediates);
    }
    return args;
  }

  poputil::graphfn::Signature GetSignature(
      const std::vector<poplar::Tensor>& args, bool training) override {
    auto signature = LstmLayerBaseOp::GetSignature(args, training);

    if (training) {
      signature.push_back(poputil::graphfn::created("intermediates"));
    }
    return signature;
  }

  void SetOutputTensor(DriverGraph& graph, std::vector<poplar::Tensor>& args,
                       const HloInstruction* inst, TensorMap& tensor_map,
                       bool training) override {
    const int64_t total_param_count = InputTensorCount() + OutputTensorCount();
    LstmLayerBaseOp::SetOutputTensor(graph, args, inst, tensor_map, training);
    if (training) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, OutputTensorCount(),
                                  DriverTensor(args[total_param_count])));
    }
  }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::lstm::LstmParams lstm_params,
                     const poplar::OptionFlags& lstm_opts,
                     const int64_t& input_size, const int64_t& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_h_state = args[1];
    poplar::Tensor input_c_state = args[2];
    poplar::Tensor kernel = args[3];
    poplar::Tensor biases = args[4];

    popnn::lstm::LstmWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackLstmKernel(kernel, input_size, output_size);
    weights.biases = biases;
    popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

    auto intermediates_ptr = training ? &args[7] : nullptr;

    std::tie(args[5], args[6]) = popnn::lstm::lstmFwd(
        graph, lstm_params, init_state, input_seq, weights, intermediates_ptr,
        prog, {debug_name_and_id}, lstm_opts, &res.planning_cache);
  }
};
REGISTER_POPLAR_OP(LstmLayerFwd, LstmLayerFwdOp);

class LstmLayerBwdOp : public LstmLayerBaseOp {
 public:
  LstmLayerBwdOp() = default;

 protected:
  const std::string ClassName() const override { return "LstmLayerBwdOp"; }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {"input_seq",
                                                 "input_h_state",
                                                 "input_c_state",
                                                 "kernel",
                                                 "bias",
                                                 "output",
                                                 "output_c_state",
                                                 "intermediates",
                                                 "output_backprop",
                                                 "output_c_state_backprop",
                                                 "input_backprop",
                                                 "input_h_state_backprop",
                                                 "input_c_state_backprop",
                                                 "kernel_backprop",
                                                 "biases_backprop"};
    return name_list;
  }

  int64_t InputTensorCount() const override { return 10; }

  int64_t OutputTensorCount() const override { return 5; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::lstm::LstmParams lstm_params,
                     const poplar::OptionFlags& lstm_opts,
                     const int64_t& input_size, const int64_t& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_h_state = args[1];
    poplar::Tensor input_c_state = args[2];
    poplar::Tensor kernel = args[3];
    poplar::Tensor biases = args[4];
    poplar::Tensor output = args[5];
    poplar::Tensor output_c_state = args[6];
    poplar::Tensor intermediates = args[7];
    poplar::Tensor output_backprop = args[8];
    poplar::Tensor output_c_state_backprop = args[9];

    popnn::lstm::LstmWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackLstmKernel(kernel, input_size, output_size);
    weights.biases = biases;

    popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

    popnn::lstm::LstmWeights weights_backprop;
    popnn::lstm::LstmState init_state_backprop = popnn::lstm::lstmBwdWithWU(
        graph, lstm_params, prog, init_state, intermediates, weights, input_seq,
        output, output_backprop, &output_c_state_backprop, &args[10],
        weights_backprop, {debug_name_and_id}, lstm_opts, &res.planning_cache);
    args[11] = init_state_backprop.output;
    args[12] = init_state_backprop.cellState;
    args[13] = PackLstmKernel(weights_backprop.inputWeights,
                              weights_backprop.outputWeights);
    args[14] = weights_backprop.biases;
  }
};
REGISTER_POPLAR_OP(LstmLayerBwd, LstmLayerBwdOp);

class DynamicLstmLayerFwdOp : public LstmLayerFwdOp {
 public:
  DynamicLstmLayerFwdOp() = default;

 protected:
  const std::string ClassName() const override {
    return "DynamicLstmLayerFwdOp";
  }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {
        "input_seq", "input_h_state", "input_c_state", "kernel",
        "bias",      "seq_len",       "output",        "output_c_state"};
    return name_list;
  }

  int64_t InputTensorCount() const override { return 6; }

  int64_t OutputTensorCount() const override { return 2; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::lstm::LstmParams lstm_params,
                     const poplar::OptionFlags& lstm_opts,
                     const int64_t& input_size, const int64_t& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_h_state = args[1];
    poplar::Tensor input_c_state = args[2];
    poplar::Tensor kernel = args[3];
    poplar::Tensor biases = args[4];
    poplar::Tensor seq_len = args[5];

    auto lstm_inst = Cast<HloDynamicLSTMFwdInstruction>(inst);
    lstm_params.preserveFinalState = lstm_inst->preserve_final_state();

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);
    lstm_params.rnn.varTimeSteps = seq_len;

    popnn::lstm::LstmWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackLstmKernel(kernel, input_size, output_size);
    weights.biases = biases;
    popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

    auto intermediates_ptr = training ? &args[8] : nullptr;

    std::tie(args[6], args[7]) = popnn::lstm::lstmFwd(
        graph, lstm_params, init_state, input_seq, weights, intermediates_ptr,
        prog, {debug_name_and_id}, lstm_opts, &res.planning_cache);
  }
};
REGISTER_POPLAR_OP(DynamicLstmLayerFwd, DynamicLstmLayerFwdOp);

class DynamicLstmLayerBwdOp : public LstmLayerBwdOp {
 public:
  DynamicLstmLayerBwdOp() = default;

 protected:
  const std::string ClassName() const override {
    return "DynamicLstmLayerBwdOp";
  }

  std::vector<const char*> NameList() const override {
    static std::vector<const char*> name_list = {"input_seq",
                                                 "input_h_state",
                                                 "input_c_state",
                                                 "kernel",
                                                 "bias",
                                                 "seq_len",
                                                 "output",
                                                 "output_c_state",
                                                 "intermediates",
                                                 "output_backprop",
                                                 "output_c_state_backprop",
                                                 "input_backprop",
                                                 "input_h_state_backprop",
                                                 "input_c_state_backprop",
                                                 "kernel_backprop",
                                                 "biases_backprop"};
    return name_list;
  }

  int64_t InputTensorCount() const override { return 11; }

  int64_t OutputTensorCount() const override { return 5; }

  void LowerToPoplar(poplar::Graph& graph, CompilerResources& res,
                     const HloInstruction* inst,
                     popnn::lstm::LstmParams lstm_params,
                     const poplar::OptionFlags& lstm_opts,
                     const int64_t& input_size, const int64_t& output_size,
                     bool training,
                     const poplar::DebugNameAndId& debug_name_and_id,
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& prog) override {
    poplar::Tensor input_seq = args[0];
    poplar::Tensor input_h_state = args[1];
    poplar::Tensor input_c_state = args[2];
    poplar::Tensor kernel = args[3];
    poplar::Tensor biases = args[4];
    poplar::Tensor seq_len = args[5];
    poplar::Tensor output = args[6];
    poplar::Tensor output_c_state = args[7];
    poplar::Tensor intermediates = args[8];
    poplar::Tensor output_backprop = args[9];
    poplar::Tensor output_c_state_backprop = args[10];

    seq_len = seq_len.reinterpret(poplar::UNSIGNED_INT);
    lstm_params.rnn.varTimeSteps = seq_len;

    popnn::lstm::LstmWeights weights;
    std::tie(weights.inputWeights, weights.outputWeights) =
        UnpackLstmKernel(kernel, input_size, output_size);
    weights.biases = biases;

    popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

    popnn::lstm::LstmWeights weights_backprop;
    popnn::lstm::LstmState init_state_backprop = popnn::lstm::lstmBwdWithWU(
        graph, lstm_params, prog, init_state, intermediates, weights, input_seq,
        output, output_backprop, &output_c_state_backprop, &args[11],
        weights_backprop, {debug_name_and_id}, lstm_opts, &res.planning_cache);
    args[12] = init_state_backprop.output;
    args[13] = init_state_backprop.cellState;
    args[14] = PackLstmKernel(weights_backprop.inputWeights,
                              weights_backprop.outputWeights);
    args[15] = weights_backprop.biases;
  }
};
REGISTER_POPLAR_OP(DynamicLstmLayerBwd, DynamicLstmLayerBwdOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
