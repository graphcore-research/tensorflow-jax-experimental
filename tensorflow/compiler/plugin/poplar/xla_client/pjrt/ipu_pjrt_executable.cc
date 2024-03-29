/* Copyright (c) 2023 Graphcore Ltd. All rights reserved.

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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_executable.h"

#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

namespace {

// A few static checks on classes, to make sure we don't need to copy around.
static_assert(std::is_nothrow_move_constructible_v<IpuPjRtRunState>);
static_assert(std::is_nothrow_move_assignable_v<IpuPjRtRunState>);
// static_assert(std::is_nothrow_move_constructible_v<IpuPjRtExecutableRunInfo>);
// static_assert(std::is_nothrow_move_assignable_v<IpuPjRtExecutableRunInfo>);
// static_assert(std::is_nothrow_move_constructible_v<IpuPjRtMeshTransition>);
// static_assert(std::is_nothrow_move_assignable_v<IpuPjRtMeshTransition>);

/**
 * @brief Check all input buffers used for streaming variables are valid (i.e.
 * synchronized with host).
 */
Status CheckInputStreamingBuffers(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const InputOutputAliasingMap& io_aliasing_map) {
  const auto& inputs_info = io_aliasing_map.GetEntryInputInfos();
  // Each replica input buffers.
  for (const auto& inputs : argument_handles) {
    CHECK_EQ(inputs_info.size(), inputs.size());
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
      // IPU buffer must be synced with host to be streamed.
      IpuPjRtBuffer* buffer =
          tensorflow::down_cast<IpuPjRtBuffer*>(inputs[idx]);
      if (inputs_info[idx].IsStreaming() && !buffer->IsHostBufferSync()) {
        return FailedPrecondition(
            "IPU streaming input requires a buffer synchronized with host.");
      }
    }
  }
  return Status::OK();
}

StatusOr<IpuPjRtBufferLocation> CheckInputDonatedBuffersLocation(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const InputOutputAliasingMap& io_aliasing_map,
    const IpuPjRtMeshTransition& mesh_transition) {
  // Check consistency of donated inputs, and compatibility with mesh
  // transition.
  bool all_buffers_on_sram = true;
  bool all_buffer_host_sync = true;
  // TODO: check also device ids?
  const auto& inputs_info = io_aliasing_map.GetEntryInputInfos();
  // Each replica input buffers.
  for (const auto& inputs : argument_handles) {
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
      IpuPjRtBuffer* buffer =
          tensorflow::down_cast<IpuPjRtBuffer*>(inputs[idx]);
      // Only looking at donated buffers.
      if (inputs_info[idx].IsResource()) {
        all_buffers_on_sram &=
            (buffer->location() == IpuPjRtBufferLocation::SRAM);
        all_buffer_host_sync &= buffer->IsHostBufferSync();
      }
    }
  }
  const bool is_ipu_sram_reset = mesh_transition.IsIpuMeshSRAMReset();
  // All buffers already on SRAM & no reset => all good!
  if (all_buffers_on_sram && !is_ipu_sram_reset) {
    return IpuPjRtBufferLocation::SRAM;
  }
  // Buffers not host sync + IPU SRAM reset => problem!
  if (is_ipu_sram_reset && !all_buffer_host_sync) {
    return FailedPrecondition(
        "The IPU device mesh with id %i will be reset (new engine load). Input "
        "SRAM buffers will be invalid. Please use host buffers as inputs.",
        mesh_transition.mesh_id);
  }
  // All buffers sync. with host => can transfer them.
  if (all_buffer_host_sync) {
    return IpuPjRtBufferLocation::HOST;
  }
  // Mix-match of buffers. Too complicated!
  return FailedPrecondition(
      "IPU PjRt client does not support a mix of HOST and SRAM buffers for "
      "donated inputs.");
}

/** @brief Get Poplar engine input stream names. */
std::unordered_set<std::string> GetEngineInputStreams(poplar::Engine* engine) {
  std::unordered_set<std::string> input_streams;
  const auto all_streams = engine->listStreams();
  for (const auto& v : all_streams) {
    if (v.back() == '+') {
      input_streams.insert(v.substr(0, v.size() - 1));
    }
  }
  return input_streams;
}
/** @brief Get Poplar engine output stream names. */
std::unordered_set<std::string> GetEngineOutputStreams(poplar::Engine* engine) {
  std::unordered_set<std::string> output_streams;
  const auto all_streams = engine->listStreams();
  for (const auto& v : all_streams) {
    if (v.back() == '-') {
      output_streams.insert(v.substr(0, v.size() - 1));
    }
  }
  return output_streams;
}

/** @brief Connecting a host callback function to a Poplar engine. */
void ConnectStreamHostCallback(const HostFunctionInfo& host_function_info,
                               poplar::Engine* engine) {
  std::vector<const void*> local_input_state(
      host_function_info.input_shapes.size());
  std::vector<void*> local_output_state(
      host_function_info.output_shapes.size());
  auto host_fn = [local_input_state, local_output_state, host_function_info](
                     poplar::ArrayRef<const void*> ins,
                     poplar::ArrayRef<void*> outs) mutable {
    absl::c_copy(ins, local_input_state.begin());
    absl::c_copy(outs, local_output_state.begin());
    host_function_info.function(local_input_state, local_output_state);
  };
  // Replica 0 only supported at the moment.
  engine->connectHostFunction(host_function_info.handle, 0, host_fn);
}
void ConnectStreamHostCallbacks(const HostFunctionInfos& host_function_infos,
                                poplar::Engine* engine) {
  for (const auto& host_function_info : host_function_infos) {
    ConnectStreamHostCallback(host_function_info.second, engine);
  }
}

/**
 * @brief Create the (optional) input host transpose plan.
 * Necessary to support any input layout for host buffers.
 */
StatusOr<std::shared_ptr<TransposePlan>> CreateInputHostTransposePlan(
    const xla::Shape& on_device_shape,
    xla::TransposePlanCache& transpose_cache) {
  // No layout => no transpose plan necessary.
  const bool has_layout = on_device_shape.has_layout();
  if (!has_layout) {
    return std::shared_ptr<TransposePlan>{nullptr};
  }
  const auto type = on_device_shape.element_type();
  const auto& dimensions = on_device_shape.dimensions();
  std::vector<int64_t> byte_strides(dimensions.size());
  // Extract bytes strides if XLA shape has layout.
  xla::ShapeUtil::ByteStrides(on_device_shape,
                              absl::Span<int64_t>(byte_strides));
  // Is it default major to minor layout?
  const bool has_default_layout =
      !has_layout || xla::HasMajorToMinorLayout(type, dimensions, byte_strides);
  if (has_default_layout) {
    // Default major to minor layout => no transpose plan necessary.
    return std::shared_ptr<TransposePlan>{nullptr};
  }
  // Not trivial layout => create (or get) host transpose plan.
  absl::InlinedVector<int64_t, 4> permutation(dimensions.size());
  absl::c_iota(permutation, 0);
  return transpose_cache.GetOrCreate(primitive_util::ByteWidth(type),
                                     dimensions, permutation,
                                     TransposePlan::Striding{byte_strides});
}

}  // namespace

/**
 * @brief IPU input stream callback.
 */
class IpuInputStreamCallback : public poplar::StreamCallback {
 public:
  IpuInputStreamCallback(std::shared_ptr<MaybeOwningCpuMemory> host_buffer,
                         std::shared_ptr<TransposePlan> host_tranpose_plan)
      : m_host_buffer{std::move(host_buffer)},
        m_host_transpose_plan{std::move(host_tranpose_plan)} {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Not yet supported.
    return poplar::StreamCallback::Result::NotAvailable;
  }
  void fetch(void* dst) noexcept override {
    // Copy from input host TFRT buffer to IPU stream buffer.
    if (m_host_transpose_plan) {
      // Not default layout => use transpose plan.
      m_host_transpose_plan->Execute(m_host_buffer->data(), dst);
    } else {
      // Major to minor => memcpy directly.
      std::memcpy(dst, m_host_buffer->data(), m_host_buffer->size());
    }
  }
  void complete() noexcept override {}

 private:
  /** Input host buffer to stream to device. */
  // TODO: support tuples?
  std::shared_ptr<MaybeOwningCpuMemory> m_host_buffer;
  /** (Optional) host transpose plan for copying the input.
   * Used if memory layout is not standard major to minor layout.
   */
  std::shared_ptr<TransposePlan> m_host_transpose_plan{nullptr};
};

/**
 * @brief IPU output stream callback.
 */
class IpuOutputStreamCallback : public poplar::StreamCallback {
 public:
  IpuOutputStreamCallback(std::shared_ptr<MaybeOwningCpuMemory> host_buffer)
      : m_host_buffer{std::move(host_buffer)} {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Not supported for outputs.
    return poplar::StreamCallback::Result::NotAvailable;
  }
  void fetch(void* src) noexcept override {
    std::memcpy(m_host_buffer->data(), src, m_host_buffer->size());
  }
  void complete() noexcept override {}

 private:
  /** Input host buffer to stream to from device. */
  // TODO: support tuples?
  std::shared_ptr<MaybeOwningCpuMemory> m_host_buffer;
};

static tfrt::AsyncValueRef<CpuEvent> GetOrCreateReadyEvent(
    tfrt::HostContext* host_context) {
  static const auto* ready_event = new tfrt::AsyncValueRef<CpuEvent>(
      tfrt::MakeAvailableAsyncValueRef<CpuEvent>(host_context));
  return ready_event->CopyRef();
}

PoplarExecutable* GetPoplarExecutable(
    PjRtStreamExecutorExecutable* executable) {
  CHECK_NOTNULL(executable);
  CHECK_EQ(executable->executables().size(), 1);
  const std::shared_ptr<LocalExecutable> local_executable =
      executable->executables()[0];
  // TODO: check the cast is valid?
  PoplarExecutable* poplar_executable =
      tensorflow::down_cast<PoplarExecutable*>(local_executable->executable());
  return poplar_executable;
}

CompileOptions CreatePoplarCompileOptions(
    const CompileOptions& compile_options,
    const IpuDeviceMeshManager& mesh_manager) {
  // Always requires device assignment for Poplar backend.
  CHECK(compile_options.executable_build_options.has_device_assignment());
  // Get associated IPU Poplar mesh.
  const auto& ipu_mesh = mesh_manager.find(
      compile_options.executable_build_options.device_assignment());

  // Adapt execute options to poplar XLA backend/compiler.
  CompileOptions poplar_compile_options = compile_options;
  ExecutableBuildOptions& poplar_build_options =
      poplar_compile_options.executable_build_options;
  // Set device id to Poplar multi-ipus id.
  poplar_build_options.set_device_ordinal(ipu_mesh.local_device_index());
  DeviceAssignment poplar_device_assign(
      poplar_build_options.device_assignment());
  poplar_device_assign.Fill(ipu_mesh.local_device_index());
  poplar_build_options.set_device_assignment(poplar_device_assign);
  return poplar_compile_options;
}

Status CheckPoplarExecutableValid(PoplarExecutable* poplar_executable,
                                  const CompileOptions& compile_options) {
  // Check all inputs/outputs are streaming.
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();
  for (const auto& v : io_aliasing_map.GetEntryInputInfos()) {
    CHECK_EQ(v.Handles().size(), 1);
    CHECK(!v.Shape().IsTuple());
    if (v.GetType() ==
        InputOutputAliasingMap::InputInfo::Type::ResourceNotModified) {
      return FailedPrecondition(
          "IPU PjRt client does not support `ResourceNotModified` inputs: "
          "`%s`.",
          v.Handles()[0]);
    }
  }
  for (const auto& v : io_aliasing_map.GetEntryOutputInfos()) {
    CHECK_EQ(v.Handles().size(), 1);
    CHECK(!v.Shape().IsTuple());
    if (v.GetType() ==
        InputOutputAliasingMap::OutputInfo::Type::ResourceOutputOnly) {
      return FailedPrecondition(
          "IPU PjRt client does not support `ResourceOutputOnly` outputs: %s.",
          v.Handles()[0]);
    }
  }
  // Infeed and outfeed not yet supported on IPU PjRt runtime.
  if (poplar_executable->GetInfeedInfos().size() > 0) {
    return FailedPrecondition(
        "Infeeds are not yet supported by IPU PjRt backend.");
  }
  if (poplar_executable->GetOutfeedInfos().size() > 0) {
    return FailedPrecondition(
        "Outfeeds are not yet supported by IPU PjRt backend.");
  }
  // Other unsupported features.
  CHECK_EQ(poplar_executable->GetStreamInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetSendInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetRecvInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetRemoteParameterInfos().size(), 0);

  // Consistency of compile options.
  CHECK(compile_options.executable_build_options.has_device_assignment());
  const DeviceAssignment& device_assignment =
      compile_options.executable_build_options.device_assignment();
  if (compile_options.executable_build_options.use_spmd_partitioning()) {
    // SPMD partitioner => XLA using partitions, not replicas.
    CHECK_EQ(device_assignment.replica_count(), 1);
    CHECK_EQ(device_assignment.computation_count(),
             poplar_executable->GetReplicationFactor());
  } else {
    // Normal JAX/XLA pmap replica config.
    CHECK_EQ(device_assignment.computation_count(), 1);
    CHECK_EQ(device_assignment.replica_count(),
             poplar_executable->GetReplicationFactor());
  }
  return Status::OK();
}

StatusOr<IpuPjRtRunReplicaInputs>
IpuPjRtRunReplicaInputs::CreateFromIpuPjRtBuffers(
    const std::vector<xla::PjRtBuffer*>& inbuffers,
    const IpuPjRtInputOutputAliasing& input_output_aliasing,
    xla::TransposePlanCache& transpose_cache) {
  IpuPjRtRunReplicaInputs inputs;
  // Highly inspired by TFRT CPU client/executable handling of buffers.
  const auto num_inputs = inbuffers.size();
  const auto& inputs_aliasing = input_output_aliasing.inputs();
  inputs.host_buffers_hold.reserve(num_inputs);
  inputs.host_tracked_buffers.reserve(num_inputs);
  inputs.host_deps.reserve(num_inputs);
  inputs.buffers_status.reserve(num_inputs);
  inputs.host_transpose_plans.reserve(num_inputs);
  // Extract host input buffers (and input events).
  for (std::size_t idx = 0; idx < num_inputs; ++idx) {
    auto* ipu_buffer = tensorflow::down_cast<IpuPjRtBuffer*>(inbuffers[idx]);
    // Not supporting buffer donation for now.
    TF_ASSIGN_OR_RETURN(
        TfrtCpuBuffer::ScopedHold host_buffer_hold,
        ipu_buffer->GetHostBufferWithHold(TfrtCpuBuffer::ScopedHold::kUsage,
                                          /*allow_unsync=*/true));
    if (!host_buffer_hold.ok()) {
      return InvalidArgument(
          "Invalid buffer passed to Execute() as argument %d: "
          "%s",
          idx, host_buffer_hold.status().ToString());
    }
    // Definition events are never modified after buffer construction.
    for (const auto& ev : host_buffer_hold->DefinitionEvents()) {
      if (!ev.IsAvailable()) {
        inputs.host_deps.push_back(ev.CopyRCRef());
      }
      // TODO: handle buffer donation.
    }
    // Mark on-device input buffer as expired for UPDATED buffers.
    // Probably unnecessary, but safer to double do it on inputs!
    if (inputs_aliasing[idx].type == IpuPjRtBufferDonationType::UPDATED) {
      ipu_buffer->status()->MarkOnDeviceExpired();
    }
    // Optional transpose plan on the input.
    TF_ASSIGN_OR_RETURN(auto input_transpose_plan,
                        CreateInputHostTransposePlan(
                            ipu_buffer->on_device_shape(), transpose_cache));
    // Not supporting multiple buffers for now.
    CHECK_EQ(host_buffer_hold->Buffers().size(), 1);
    inputs.host_tracked_buffers.push_back(host_buffer_hold.buffer());
    inputs.host_buffers_hold.emplace_back(std::move(host_buffer_hold));
    inputs.buffers_status.push_back(ipu_buffer->status());
    inputs.host_transpose_plans.push_back(std::move(input_transpose_plan));
  }
  return inputs;
}

void IpuPjRtRunReplicaInputs::ConnectStreamCallbacks(
    const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
    int replica, poplar::Engine* engine) {
  const auto num_inputs = input_infos.size();
  for (std::size_t i = 0; i < num_inputs; ++i) {
    const auto& ininfo = input_infos[i];
    const auto& inname = ininfo.Handles()[0];
    if (ininfo.IsStreaming()) {
      // TODO: pass directly raw host buffer pointer (instead of shared ptr)?
      // TODO: support tuple input?
      engine->connectStreamToCallback(
          inname, replica,
          std::make_unique<IpuInputStreamCallback>(
              host_tracked_buffers[i]->Buffers()[0], host_transpose_plans[i]));
    }
  }
}

void IpuPjRtRunReplicaInputs::ConnectH2DStreamDonatedBuffers(
    const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
    int replica, poplar::Engine* engine) {
  const auto num_inputs = input_infos.size();
  for (std::size_t i = 0; i < num_inputs; ++i) {
    const auto& ininfo = input_infos[i];
    const auto& inname = ininfo.Handles()[0];
    if (ininfo.IsResource()) {
      // Connect host buffer to stream to SRAM.
      const auto& inbuffer = host_tracked_buffers[i]->Buffers()[0];
      engine->connectStreamToCallback(inname, replica,
                                      std::make_unique<IpuInputStreamCallback>(
                                          inbuffer, host_transpose_plans[i]));
    }
  }
}

void IpuPjRtRunReplicaInputs::ConvertBufferHold(
    tfrt::AsyncValueRef<CpuEvent> execute_event) {
  for (TfrtCpuBuffer::ScopedHold& b : this->host_buffers_hold) {
    if (b.type() == TfrtCpuBuffer::ScopedHold::kUsage) {
      // Convert usage hold into usage event (using execute event).
      std::array<tfrt::AsyncValueRef<CpuEvent>, 1> usage_events{
          execute_event.CopyRef()};
      b.ConvertUsageHold(absl::MakeSpan(usage_events));
    } else {
      // TODO: handle host buffer donation.
      // CHECK(b.type() == TfrtCpuBuffer::ScopedHold::kDonation);
      // b.ConfirmDonation();
    }
  }
  // Clear the collection of buffer holds.
  this->host_buffers_hold.clear();
}

StatusOr<IpuPjRtRunReplicaOutputs>
IpuPjRtRunReplicaOutputs::AllocateFromOutputInfos(
    const tfrt::AsyncValueRef<CpuEvent>& execute_event,
    const IpuPjRtRunReplicaInputs& inputs,
    const IpuPjRtInputOutputAliasing& input_output_aliasing,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos) {
  IpuPjRtRunReplicaOutputs outputs;
  // TODO: support tuple outputs?
  // TODO: improve alllocation to deal with memory fragmentation?
  const std::size_t num_outputs = output_infos.size();
  outputs.host_tracked_buffers.reserve(num_outputs);
  for (std::size_t idx = 0; idx < output_infos.size(); ++idx) {
    const auto& output_info = output_infos[idx];
    const auto& output_aliasing = input_output_aliasing.outputs()[idx];
    // Aliasing keep a copy of the shape. TODO: do we really need it?
    outputs.shapes.push_back(output_info.Shape());
    if (output_aliasing.type == IpuPjRtBufferDonationType::UNCHANGED) {
      // `UNCHANGED` buffer => can keep input host buffer.
      // WARNING: status needs to also be shared when creating output PjRt
      // buffer!
      const auto inidx = output_aliasing.input_index;
      outputs.host_tracked_buffers.push_back(
          inputs.host_tracked_buffers[inidx]);
    } else {
      // TODO: support buffer donation for `UPDATED` buffers.
      // Allocate proper raw host buffer.
      TF_ASSIGN_OR_RETURN(auto out_buffer,
                          CreateRawHostBuffer(output_info.Shape()));
      // Wrap the raw buffer into a tracker host buffer.
      absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> sub_buffers;
      sub_buffers.push_back(std::move(out_buffer));
      // Program execution writes to output buffers so it's a definition event.
      absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
      definition_events.push_back(execute_event.CopyRef());
      // TRFT buffer wrapping raw buffers and events.
      auto out_tracked_buffer = std::make_shared<TrackedTfrtCpuDeviceBuffer>(
          /*is_tuple=*/false, std::move(sub_buffers),
          std::move(definition_events));
      outputs.host_tracked_buffers.push_back(out_tracked_buffer);
    }
  }
  return outputs;
}

std::vector<std::unique_ptr<PjRtBuffer>>
IpuPjRtRunReplicaOutputs::CreateIpuPjRtBuffers(
    const tfrt::AsyncValueRef<CpuEvent>& execute_event,
    const std::vector<std::shared_ptr<IpuPjRtBufferStatus>>&
        input_buffers_status,
    const IpuPjRtInputOutputAliasing& input_output_aliasing,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    PjRtDevice* ipu_device, IpuPjRtExecutable* executable) const {
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  // Create wrapping PjRt output buffers.
  const auto num_outputs = this->size();
  CHECK_EQ(num_outputs, output_infos.size());
  buffers.reserve(num_outputs);
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    // IPU buffer location from the IO aliasing map information.
    const auto& output_aliasing = input_output_aliasing.outputs()[idx];
    if (output_aliasing.type == IpuPjRtBufferDonationType::UNCHANGED) {
      // UNCHANGED SRAM buffer => keep the same shared "status" as input.
      auto status = input_buffers_status[output_aliasing.input_index];
      auto out_buffer = IpuPjRtBuffer::CreateIpuBuffer(
          this->shapes[idx], this->host_tracked_buffers[idx], status,
          ipu_device, executable);
      buffers.push_back(std::move(out_buffer));
    } else {
      // UPDATED or STREAMED buffer. TODO: handle donation properly?
      const auto& outinfo = output_infos[idx];
      const auto location = IpuBufferLocationFromIOType(outinfo.GetType());
      // Wrapping everything into an IPU buffer!
      auto out_buffer = IpuPjRtBuffer::CreateIpuBuffer(
          this->shapes[idx], this->host_tracked_buffers[idx], location,
          ipu_device, executable);
      buffers.push_back(std::move(out_buffer));
    }
  }
  return buffers;
}

void IpuPjRtRunReplicaOutputs::ConnectStreamCallbacks(
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    int replica, poplar::Engine* engine) {
  const auto num_outputs = output_infos.size();
  const auto engine_output_streams = GetEngineOutputStreams(engine);
  for (std::size_t i = 0; i < num_outputs; ++i) {
    const auto& outinfo = output_infos[i];
    // Connect only streamed outputs.
    if (outinfo.IsStreaming()) {
      const auto& outname = outinfo.Handles()[0];
      if (engine_output_streams.count(outname) == 0) {
        LOG_FIRST_N(WARNING, 1) << absl::StrFormat(
            "Ignoring XLA streaming output '%s'. No associated Poplar output "
            "stream.",
            outname);
      } else {
        // TODO: support tuples properly.
        engine->connectStreamToCallback(
            outname, replica,
            std::make_unique<IpuOutputStreamCallback>(
                host_tracked_buffers[i]->Buffers()[0]));
      }
    }
  }
}

IpuPjRtRunState::~IpuPjRtRunState() {
  if (bool(execute_event) && !execute_event.IsAvailable()) {
    // Set the event in error state, to avoid blocking.
    execute_event.SetError(
        absl::StrFormat("IPU executable (id: %i) run %i cancelled.",
                        run_info.executable_id, run_info.run_id));
  }
}
IpuPjRtRunState::IpuPjRtRunState(IpuPjRtRunState&& rhs) noexcept
    : run_info{std::move(rhs.run_info)},
      mesh_transition{std::move(rhs.mesh_transition)},
      all_inputs{std::move(rhs.all_inputs)},
      all_outputs{std::move(rhs.all_outputs)},
      execute_event{std::move(rhs.execute_event)},
      random_seed{std::move(rhs.random_seed)},
      inputs_donated_location{std::move(rhs.inputs_donated_location)} {
  // Manual implementation. tfrt::RCReference not movable_noexcept
}
IpuPjRtRunState& IpuPjRtRunState::operator=(IpuPjRtRunState&& rhs) noexcept {
  // Manual implementation. tfrt::RCReference not movable_noexcept
  run_info = std::move(rhs.run_info);
  mesh_transition = std::move(rhs.mesh_transition);
  all_inputs = std::move(rhs.all_inputs);
  all_outputs = std::move(rhs.all_outputs);
  execute_event = std::move(rhs.execute_event);
  random_seed = std::move(rhs.random_seed);
  inputs_donated_location = std::move(rhs.inputs_donated_location);
  return *this;
}

StatusOr<std::unique_ptr<IpuPjRtRunState>> IpuPjRtRunState::CreateWithIOBuffers(
    tfrt::AsyncValueRef<CpuEvent> execute_event,
    absl::Span<const std::vector<PjRtBuffer*>> all_input_handles,
    const IpuPjRtInputOutputAliasing& input_output_aliasing,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    xla::TransposePlanCache& transpose_cache) {
  const auto num_replicas = all_input_handles.size();
  auto run_state = std::make_unique<IpuPjRtRunState>();
  // Copy the execute event into the state.
  run_state->execute_event = std::move(execute_event);
  run_state->all_inputs.reserve(num_replicas);
  run_state->all_outputs.reserve(num_replicas);
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    // Input buffers for the replica.
    TF_ASSIGN_OR_RETURN(auto inputs,
                        IpuPjRtRunReplicaInputs::CreateFromIpuPjRtBuffers(
                            all_input_handles[replica], input_output_aliasing,
                            transpose_cache));
    // Raw output buffers for the replica.
    TF_ASSIGN_OR_RETURN(auto outputs,
                        IpuPjRtRunReplicaOutputs::AllocateFromOutputInfos(
                            run_state->execute_event, inputs,
                            input_output_aliasing, output_infos));
    run_state->all_inputs.push_back(std::move(inputs));
    run_state->all_outputs.push_back(std::move(outputs));
  }
  return run_state;
}

void IpuPjRtRunState::ConnectStreamCallbacks(
    const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    poplar::Engine* engine) {
  const auto num_replicas = this->num_replicas();
  // Connect streams from all replicas.
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    all_inputs[replica].ConnectStreamCallbacks(input_infos, replica, engine);
    all_outputs[replica].ConnectStreamCallbacks(output_infos, replica, engine);
  }
  // Random seed initialize with constant on IPU, no streaming.
  // See: PoplarXlaFlags::stream_random_seed
}

void IpuPjRtRunState::ConnectH2DStreamDonatedBuffers(
    const std::vector<InputOutputAliasingMap::InputInfo>& input_infos,
    poplar::Engine* engine) {
  const auto num_replicas = this->num_replicas();
  // Connect streams from all replicas.
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    all_inputs[replica].ConnectH2DStreamDonatedBuffers(input_infos, replica,
                                                       engine);
  }
}

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtRunState::CreateOutputIpuPjRtBuffers(
    const tfrt::AsyncValueRef<CpuEvent>& execute_event,
    const IpuPjRtInputOutputAliasing& input_output_aliasing,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    std::vector<PjRtDevice*> ipu_devices, IpuPjRtExecutable* executable) const {
  const auto num_replicas = this->num_replicas();
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> all_pjrt_outputs;
  all_pjrt_outputs.reserve(num_replicas);
  // Create IPU PjRt buffers wrapping raw outputs.
  for (std::size_t idx = 0; idx < num_replicas; ++idx) {
    all_pjrt_outputs.push_back(all_outputs[idx].CreateIpuPjRtBuffers(
        execute_event, all_inputs[idx].buffers_status, input_output_aliasing,
        output_infos, ipu_devices[idx], executable));
  }
  // Common run reference to all outputs, for handling on-device buffers.
  // TODO: handle returned status?
  IpuPjRtRunOutputsRef::CreateAndAssign(all_pjrt_outputs, executable);
  return all_pjrt_outputs;
}

void IpuPjRtRunState::ConvertInputBufferHold() {
  const auto num_replicas = this->num_replicas();
  for (std::size_t idx = 0; idx < num_replicas; ++idx) {
    all_inputs[idx].ConvertBufferHold(this->execute_event);
  }
}

StatusOr<std::shared_ptr<IpuPjRtRunOutputsRef>>
IpuPjRtRunOutputsRef::CreateAndAssign(
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>&
        all_out_buffers,
    IpuPjRtExecutable* executable) {
  CHECK_NOTNULL(executable);
  CHECK_GT(all_out_buffers.size(), 0);
  // Executable outputs aliasing info.
  const auto& outputs_aliasing = executable->input_output_aliasing().outputs();

  auto outref = std::make_shared<IpuPjRtRunOutputsRef>();
  outref->executable = executable;
  outref->output_buffers.resize(all_out_buffers.size());
  // Extract buffer pointer and status shared ptr.
  for (size_t r = 0; r < all_out_buffers.size(); ++r) {
    const auto& out_buffers = all_out_buffers[r];
    outref->output_buffers[r].reserve(out_buffers.size());
    for (std::size_t idx = 0; idx < out_buffers.size(); ++idx) {
      IpuPjRtBuffer* ipu_buffer =
          tensorflow::down_cast<IpuPjRtBuffer*>(out_buffers[idx].get());
      CHECK_NOTNULL(ipu_buffer);
      // Keep output donation type for expiration logic.
      outref->output_buffers[r].push_back(BufferAndStatusPtrs{
          ipu_buffer, ipu_buffer->status(), outputs_aliasing[idx]});
    }
  }
  // Assign IPU run outputs ref to all PjRt buffers.
  for (const auto& out_buffers : all_out_buffers) {
    for (const auto& buffer : out_buffers) {
      IpuPjRtBuffer* ipu_buffer =
          tensorflow::down_cast<IpuPjRtBuffer*>(buffer.get());
      ipu_buffer->AssignRunOutputsRef(outref);
    }
  }
  return outref;
}

void IpuPjRtRunOutputsRef::MarkOnDeviceExpired(
    bool keep_unchanged_donated_buffers) {
  for (auto& outbuffers_ref : output_buffers) {
    for (auto& outbuf_ref : outbuffers_ref) {
      // Should we keep unchanged donated buffers?
      if (!keep_unchanged_donated_buffers ||
          outbuf_ref.aliasing.type != IpuPjRtBufferDonationType::UNCHANGED) {
        outbuf_ref.status->MarkOnDeviceExpired();
      }
    }
  }
}

bool IpuPjRtRunOutputsRef::IsAnyOnDeviceExpired() const noexcept {
  bool is_expired = false;
  // IsOnDeviceExpired
  for (auto& outbuffers_ref : output_buffers) {
    for (auto& outbuf_ref : outbuffers_ref) {
      const auto& status = outbuf_ref.status;
      // Only caring about SRAM buffers.
      is_expired |= (status->IsOnDeviceExpired() &&
                     status->location() == IpuPjRtBufferLocation::SRAM);
    }
  }
  return is_expired;
}

std::string IpuPjRtInputOutputDonationInfo::ToString() const noexcept {
  if (type == IpuPjRtBufferDonationType::NONE) {
    // Empty for input without donation.
    return std::string();
  } else if (type == IpuPjRtBufferDonationType::UNCHANGED) {
    return absl::StrFormat("%i:%i (UNCHANGED)", input_index, output_index);
  } else {
    return absl::StrFormat("%i:%i (UPDATED)", input_index, output_index);
  }
}

/**
 * @brief Basic analysis of HLO module to find inputs returned as such.
 */
std::vector<int> FindUnchangedInputsOutputs(HloModule* hlo_module) {
  HloComputation* entry_computation = hlo_module->entry_computation();
  HloInstruction* root_instruction = entry_computation->root_instruction();

  const std::size_t num_params = entry_computation->num_parameters();
  // By default: no input returned unchanged.
  auto inputs_unchanged = std::vector<int>(num_params, -1);
  // No root tuple instruction? I guess there is no unchanged input/output then!
  // Does not make sense to return a single output which is an unchanged input!
  if (root_instruction->opcode() != HloOpcode::kTuple) {
    return inputs_unchanged;
  }

  const auto& inputs_inst = entry_computation->parameter_instructions();
  const auto& outputs_inst = root_instruction->operands();
  for (std::size_t idx = 0; idx < num_params; ++idx) {
    // Find matching output instruction.
    // TODO: is it brittle? Do we need something more robust?
    const auto it =
        std::find(outputs_inst.begin(), outputs_inst.end(), inputs_inst[idx]);
    if (it != outputs_inst.end()) {
      inputs_unchanged[idx] = std::distance(outputs_inst.begin(), it);
    }
  }
  return inputs_unchanged;
}

IpuPjRtInputOutputAliasing::IpuPjRtInputOutputAliasing() {}
IpuPjRtInputOutputAliasing::IpuPjRtInputOutputAliasing(
    std::vector<IpuPjRtInputOutputDonationInfo> inputs_aliasing,
    std::size_t num_outputs)
    : m_inputs_aliasing{std::move(inputs_aliasing)} {
  // Build associated outputs aliasing collection.
  m_outputs_aliasing.resize(num_outputs);
  for (const auto& input_alias : m_inputs_aliasing) {
    if (input_alias.output_index >= 0) {
      m_outputs_aliasing[input_alias.output_index] =
          IpuPjRtInputOutputDonationInfo{input_alias.type,
                                         input_alias.input_index,
                                         input_alias.output_index};
    }
  }
}

IpuPjRtInputOutputAliasing IpuPjRtInputOutputAliasing::CreateFromBaseExecutable(
    PjRtExecutable* base_executable) {
  const bool is_host_executable =
      base_executable->client()->platform_id() == CpuId();
  if (is_host_executable) {
    // Empty for now. Not required when we run on CPU.
    return IpuPjRtInputOutputAliasing();
  }
  // Extract TF backend poplar executable info.
  auto poplar_executable = GetPoplarExecutable(
      reinterpret_cast<PjRtStreamExecutorExecutable*>(base_executable));
  const auto& tf_io_aliasing_map =
      poplar_executable->GetInputOutputAliasingMap();

  const auto& inputs_info = tf_io_aliasing_map.GetEntryInputInfos();
  std::vector<IpuPjRtInputOutputDonationInfo> inputs_aliasing;
  inputs_aliasing.reserve(inputs_info.size());
  for (std::size_t idx = 0; idx < inputs_info.size(); ++idx) {
    const auto& ininfo = inputs_info[idx];
    if (ininfo.IsResource()) {
      // Updated donated input by default.
      inputs_aliasing.push_back(IpuPjRtInputOutputDonationInfo{
          IpuPjRtBufferDonationType::UPDATED, int(idx),
          int(ininfo.GetOutputIndex())});
    } else {
      // No input donation.
      inputs_aliasing.push_back(
          IpuPjRtInputOutputDonationInfo{IpuPjRtBufferDonationType::NONE});
    }
  }
  // Find unchanged inputs/outputs in HLO module.
  const auto hlo_module = base_executable->GetHloModules().value()[0];
  const auto unchanged_inouts = FindUnchangedInputsOutputs(hlo_module.get());
  CHECK_EQ(inputs_info.size(), unchanged_inouts.size());
  for (std::size_t idx = 0; idx < inputs_info.size(); ++idx) {
    auto& io_alias_info = inputs_aliasing[idx];
    // Ignore `UNCHANGED` inout not marked as donated in HLO.
    if (unchanged_inouts[idx] >= 0 &&
        io_alias_info.type != IpuPjRtBufferDonationType::NONE) {
      // Updated IO alias type to `UNCHANGED`.
      io_alias_info.type = IpuPjRtBufferDonationType::UNCHANGED;
      // Got a problem if there is a mismatch here!
      CHECK_EQ(io_alias_info.output_index, unchanged_inouts[idx]);
    }
  }
  const std::size_t num_outputs =
      tf_io_aliasing_map.GetEntryOutputInfos().size();
  return IpuPjRtInputOutputAliasing(std::move(inputs_aliasing), num_outputs);
}

std::string IpuPjRtInputOutputAliasing::ToString() const noexcept {
  std::string s = "{";
  // Awful way of generating a string, but not a big performance issue here.
  for (const auto& info : m_inputs_aliasing) {
    std::string info_str = info.ToString();
    if (info_str.size() > 0) {
      s += info_str + ", ";
    }
  }
  s += "}";
  return s;
}

/////////////////////////////////////////////////////////////////////////////////////////
IpuPjRtExecutable::IpuPjRtExecutable(
    bool asynchronous, int64_t executable_id,
    std::unique_ptr<PjRtStreamExecutorExecutable> ipu_se_executable,
    std::unique_ptr<TfrtCpuExecutable> host_executable,
    const CompileOptions& compile_options, IpuPjRtClient* client)
    : m_asynchronous_run{asynchronous},
      m_executable_id{executable_id},
      m_ipu_se_executable{std::move(ipu_se_executable)},
      m_host_executable{std::move(host_executable)},
      m_compile_options{compile_options},
      m_client{client} {
  // Should have at least one or the other!
  CHECK(bool(m_ipu_se_executable) || bool(m_host_executable));

  const auto& all_devices = client->addressable_devices();
  const std::vector<int> device_ids(device_assignment().begin(),
                                    device_assignment().end());
  // Build the collection of addressable devices for the executable.
  m_devices.clear();
  m_devices.reserve(device_ids.size());
  m_addressable_device_logical_ids.reserve(device_ids.size());
  for (std::size_t idx = 0; idx < device_ids.size(); ++idx) {
    const int device_id = device_ids[idx];
    auto it =
        std::find_if(all_devices.begin(), all_devices.end(),
                     [device_id](auto v) { return v->id() == device_id; });
    m_devices.push_back(*it);
    m_addressable_device_logical_ids.push_back(
        PjRtExecutable::LogicalDeviceIds{int(idx), 0});
  }
  // IPU device mesh associated.
  m_device_mesh_id =
      m_client->ipu_mesh_manager().find(device_assignment()).id();
  // A few checks!
  CHECK_GT(m_devices.size(), 0);
  CHECK_NOTNULL(GetBaseExecutable());
  CHECK(m_compile_options.executable_build_options.has_device_assignment());

  // In/out aliasing info.
  m_input_output_aliasing =
      IpuPjRtInputOutputAliasing::CreateFromBaseExecutable(GetBaseExecutable());
  LOG(INFO) << "IPU PjRt executable input/output aliasing map: "
            << m_input_output_aliasing.ToString();

  // Start execute + cleanup threads in asynchronous case.
  if (m_asynchronous_run) {
    m_execute_thread = std::thread(&IpuPjRtExecutable::ExecuteAsyncLoop, this);
    m_cleanup_thread =
        std::thread(&IpuPjRtExecutable::ExecuteCleanupAsyncLoop, this);
  }
}
IpuPjRtExecutable::~IpuPjRtExecutable() {
  TENSORFLOW_TRACEPOINT();
  this->Delete();
}

PjRtClient* IpuPjRtExecutable::client() const { return m_client; }
// Unique name for this executable, e.g., HloModule name.
absl::string_view IpuPjRtExecutable::name() const {
  return GetBaseExecutable()->name();
}
int IpuPjRtExecutable::num_replicas() const {
  return device_assignment().replica_count();
}
int IpuPjRtExecutable::num_partitions() const {
  return device_assignment().computation_count();
}
int64_t IpuPjRtExecutable::SizeOfGeneratedCodeInBytes() const {
  throw std::runtime_error(
      "Not implemented `SizeOfGeneratedCodeInBytes` on IPU.");
}

const DeviceAssignment& IpuPjRtExecutable::device_assignment() const {
  return m_compile_options.executable_build_options.device_assignment();
}
absl::Span<const PjRtExecutable::LogicalDeviceIds>
IpuPjRtExecutable::addressable_device_logical_ids() const {
  return m_addressable_device_logical_ids;
}
absl::Span<PjRtDevice* const> IpuPjRtExecutable::addressable_devices() const {
  return m_devices;
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
IpuPjRtExecutable::GetHloModules() const {
  return GetBaseExecutable()->GetHloModules();
}

// Executes on devices addressable by the client. Requires executable has a
// device_assignment and all devices in the device_assignment are addressable
// by the client.
//
// `argument_handles` is `[num_devices, num_args]`.
//
// If returned_futures.has_value():
//   if Execute does not return an error status:
//     *returned_futures will be resized to be the same length as the return
//     vector, and each future will become ready once the corresponding device
//     execute has completed.
//   else:
//     *returned_futures is undefined.
//
// The caller is *NOT* required to ensure that PjRtExecutable stays alive
// until futures are ready.
StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
IpuPjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  TENSORFLOW_TRACEPOINT();

  // Forward call directly to host executable.
  if (UseHostExecutable()) {
    return ExecuteOnHost(argument_handles, options, returned_futures);
  }
  LOG(INFO) << "Creating input/output buffers for executable run: " << name();

  auto* host_context = m_client->cpu_client()->GetHostContext();
  // Poplar executable and associated in/out mapping.
  CHECK_NOTNULL(m_ipu_se_executable.get());
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();

  const std::size_t num_inputs = io_aliasing_map.GetEntryInputInfos().size();
  const std::size_t num_outputs = io_aliasing_map.GetEntryOutputInfos().size();
  // A couple of early checks on arguments!
  const std::size_t num_addressable_devices = m_devices.size();
  if (argument_handles.size() != num_addressable_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_addressable_devices, num_replicas(),
        num_partitions());
  }
  for (const auto& arguments_device : argument_handles) {
    CHECK_EQ(arguments_device.size(), num_inputs);
    TF_RETURN_IF_ERROR(ValidateArgumentHandles(arguments_device));
  }
  // Estimate of the mesh transition given existing client state.
  const auto mesh_transition =
      m_client->EstimateClientMeshTransition(m_device_mesh_id, m_executable_id);
  // Check streaming input buffers.
  TF_RETURN_IF_ERROR(
      CheckInputStreamingBuffers(argument_handles, io_aliasing_map));
  // Check on device/donated input buffers, and get the location of these input
  // buffers. On HOST location requires the buffers to be first streamed before
  // calling the main program.
  TF_ASSIGN_OR_RETURN(const IpuPjRtBufferLocation inputs_donated_location,
                      CheckInputDonatedBuffersLocation(
                          argument_handles, io_aliasing_map, mesh_transition));

  // execute_event indicates whether ipu computation is complete and whether
  // there was an error.
  // Different type of execute event for sync/async cases.
  tfrt::AsyncValueRef<CpuEvent> execute_event =
      m_asynchronous_run ? tfrt::MakeConstructedAsyncValueRef<CpuEvent>(
                               host_context)                     // Asynchronous
                         : GetOrCreateReadyEvent(host_context);  // Synchronous

  // PjRt run state with all replicas IO buffers.
  TF_ASSIGN_OR_RETURN(
      auto run_state,
      IpuPjRtRunState::CreateWithIOBuffers(
          execute_event, argument_handles, m_input_output_aliasing,
          io_aliasing_map.GetEntryOutputInfos(), m_host_transpose_cache));
  run_state->inputs_donated_location = inputs_donated_location;

  // Wrapping execute even as PjRt future status.
  auto done_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
  run_state->execute_event.AndThen(
      [done_event = done_event.CopyRef(),
       event = run_state->execute_event.CopyRef()]() {
        Status s;
        if (auto* error = event.GetErrorIfPresent()) {
          s = InternalError("Compute error: %s", error->message);
        }
        done_event.emplace(std::move(s));
      });
  auto future_status = PjRtFuture<Status>(std::move(done_event));

  // Returned outputs, for all replica, with IPU executable reference.
  auto outputs = run_state->CreateOutputIpuPjRtBuffers(
      run_state->execute_event, this->input_output_aliasing(),
      io_aliasing_map.GetEntryOutputInfos(), m_devices, this);
  // Returned futures, for all replica.
  if (returned_futures) {
    returned_futures.value().clear();
    for (std::size_t idx = 0; idx < run_state->num_replicas(); ++idx) {
      returned_futures.value().push_back(future_status);
    }
  }
  // Run outputs reference. Implicitely assuming all output buffers have the
  // same ref. Keep it in case the executable gets deleted.
  CHECK_GT(outputs[0].size(), 0);
  auto m_last_run_outputs_ref =
      tensorflow::down_cast<IpuPjRtBuffer*>(outputs[0][0].get())
          ->run_outputs_ref();

  // Poplar engine: if null, means it should be executed on HOST directly
  // (constant, scalar, ...). NOTE: check engine after inputs status.
  poplar::Engine* engine = poplar_executable->Engine();
  CHECK_NOTNULL(engine);

  // (Blocking) update of client state.
  // Use as coordination mechanism when required to load or reorganize IPUs.
  // NOTE: passing inputs donated location, as used for marking previous SRAM
  // (UNCHANGED or UPDATED) buffers as expired (or not).
  auto [update_run_info, update_mesh_transition] = m_client->UpdateClientState(
      m_device_mesh_id, m_executable_id, run_state->inputs_donated_location,
      m_last_run_outputs_ref, run_state->execute_event.CopyRef());
  // Move run info and mesh transition info.
  run_state->run_info = std::move(update_run_info);
  run_state->mesh_transition = std::move(update_mesh_transition);

  // No need for inputs scoped hold => convert to usage event.
  // NOTE: necessary, otherwise scoped hold blocking buffer delete.
  run_state->ConvertInputBufferHold();
  // Mark unchanged donated input buffers as SRAM synchronized.
  // Allows to write nice JAX inference code, where the donated output is
  // ignored and weights are on SRAM for next iteration.
  this->MarkUnchangedDonatedBuffersAsSynchronizedOnSRAM(argument_handles);

  LOG(INFO) << "Queuing IPU computation " << name()
            << "; num_replicas=" << num_replicas()
            << " num_partitions=" << num_partitions()
            << "; num_addressable_devices=" << num_addressable_devices
            << "; num_inputs=" << num_inputs << " num_outputs=" << num_outputs
            << "; executable id: " << run_state->run_info.executable_id
            << "; run id: " << run_state->run_info.run_id
            << "; mesh_id: " << m_device_mesh_id;

  /////////////// RUNNING POPLAR ENGINE ///////////////
  /////////////// RUNNING POPLAR ENGINE ///////////////
  /////////////// RUNNING POPLAR ENGINE ///////////////
  if (m_asynchronous_run) {
    // Queue the state, for async. Poplar engine run.
    m_execute_run_state_queue.enqueue(std::move(run_state));
  } else {
    // Synchronous Poplar engine run.
    this->ExecuteDeviceRun(*run_state);
  }
  return outputs;
}

void IpuPjRtExecutable::ExecuteDeviceRun(IpuPjRtRunState& run_state) {
  TENSORFLOW_TRACEPOINT();
  // Single use of Poplar::Engine at a time!
  std::scoped_lock l(m_poplar_engine_mutex);
  // Poplar executable and associated in/out mapping.
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();
  poplar::Engine* engine = poplar_executable->Engine();

  // Blocking events, from same mesh (or overlapping ones).
  tfrt::Await(run_state.mesh_transition.mesh_blocking_events);
  if (run_state.mesh_transition.require_device_attach) {
    // Make sure the proper IPU Poplar mesh is attached.
    // Forcing detaching any other mesh overlapping.
    // TODO: proper wait for other IPU meshes?
    m_client->ipu_mesh_manager().Attach(m_device_mesh_id, true);
  }
  if (run_state.mesh_transition.require_engine_load) {
    const auto& device_mesh =
        m_client->ipu_mesh_manager().find(m_device_mesh_id);
    LOG(INFO) << "Load IPU poplar engine " << name()
              << ", executable id: " << run_state.run_info.executable_id
              << ", on Poplar device with ID: " << m_device_mesh_id;
    // First call of the executable => load engine.
    engine->load(device_mesh.device());
  }
  // Check input buffer status?
  //   for (const auto& av : input_deps_avs) {
  //   if (auto* error = av->GetErrorIfPresent()) {
  //     execute_event.SetError(absl::StrCat(
  //         "Error dispatching computation: %s", error->message));
  //     return;
  //   }
  // }

  // Wait for all inputs to be ready?
  for (std::size_t idx = 0; idx < run_state.num_replicas(); ++idx) {
    tfrt::Await(run_state.all_inputs[idx].host_deps);
  }
  // Transfer donated input buffers to IPU.
  if (run_state.inputs_donated_location == IpuPjRtBufferLocation::HOST) {
    LOG(INFO) << "Transfer H2D buffers from HOST to IPU:" << name()
              << "; executable id: " << run_state.run_info.executable_id
              << "; run id: " << run_state.run_info.run_id;
    // Connect on-device buffers H2D streams.
    run_state.ConnectH2DStreamDonatedBuffers(
        io_aliasing_map.GetEntryInputInfos(), engine);
    engine->run(PoplarProgramType::HOST_TO_DEVICE);
  }
  // Connect all replicas streams for running main program.
  run_state.ConnectStreamCallbacks(io_aliasing_map.GetEntryInputInfos(),
                                   io_aliasing_map.GetEntryOutputInfos(),
                                   engine);
  // Connect engine streams corresponding to host callbacks.
  ConnectStreamHostCallbacks(poplar_executable->GetHostFunctionInfos(), engine);

  // Synchronous call => blocking thread!
  LOG(INFO) << "Run IPU poplar engine " << name()
            << "; executable id: " << run_state.run_info.executable_id
            << "; run id: " << run_state.run_info.run_id;
  engine->run(PoplarProgramType::MAIN_SEQUENCE);
}

void IpuPjRtExecutable::ExecuteAsyncLoop() {
  while (!this->IsDeleted()) {
    // Wait for the next run state.
    auto run_state = m_execute_run_state_queue.dequeue();
    // Ignore empty/nulltpr run state => break loop.
    if (!run_state) {
      continue;
    }
    // Synchronous/blocking Poplar engine run.
    this->ExecuteDeviceRun(*run_state);
    // Mark the execute event as done!
    run_state->execute_event.SetStateConcrete();
    // Move the run state to the cleanup queue.
    // Why?! deleting the run state is non negligible overhead in the loop!
    m_clean_run_state_queue.enqueue(std::move(run_state));
  }
}

void IpuPjRtExecutable::ExecuteCleanupAsyncLoop() {
  while (!this->IsDeleted()) {
    // Wait for the next state to cleanup!
    auto run_state = m_clean_run_state_queue.dequeue();
    TENSORFLOW_TRACEPOINT();
    // Ignore empty/nulltpr run state => break loop.
    if (!run_state) {
      continue;
    }
    // Explicit reset (not really necessary!)
    run_state.reset();
  }
}

// Execute the assigned replica/partition on a given `device`. Requires
// executable has a device_assignment, `device` is present in the
// device_assignment and addressable by the client.
//
// If fill_future is true:
//   if ExecuteSharded does not return an error status:
//     returned_future will be filled with a future that will become ready
//     once the execution has completed.
//    else:
//     returned_future will not be modified.
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  throw std::runtime_error("Not implemented `ExecuteSharded` on IPU.");
}

// Execute on a given `device`. Requires `device` to be addressable by client.
// Requires executable has exactly 1 replica and 1 partition and no
// device_assignment (thus portable).
//
// If fill_future is true:
//   if ExecutePortable does not return an error status:
//     returned_future will be filled with a future that will become ready
//     once the execution has completed.
//    else:
//     returned_future will not be modified.
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  throw std::runtime_error("Not implemented `ExecutePortable` on IPU.");
}

// Asynchronously free resources after the last execution completes.
void IpuPjRtExecutable::Delete() {
  m_executable_is_deleted.store(true);
  if (m_asynchronous_run) {
    // Enqueue nullptr, to unblock threads.
    m_execute_run_state_queue.enqueue(nullptr);
    m_clean_run_state_queue.enqueue(nullptr);
    m_execute_thread.join();
    m_cleanup_thread.join();
  }
  // Mark last run on-device expired, and nullify ptr.
  // Avoid error if trying to fetch on-device values.
  if (m_last_run_outputs_ref) {
    const bool keep_unchanged_donated_buffers = false;
    m_last_run_outputs_ref->MarkOnDeviceExpired(keep_unchanged_donated_buffers);
    m_last_run_outputs_ref->executable = nullptr;
  }
  // Reset host and IPU executables.
  m_ipu_se_executable.reset();
  m_host_executable.reset();
}

// True if on-device resources associated with the executable are freed.
bool IpuPjRtExecutable::IsDeleted() { return m_executable_is_deleted.load(); }

const poplar::Device& IpuPjRtExecutable::GetPoplarDevice() const {
  CHECK_GE(m_device_mesh_id, 0);
  const auto& mesh = m_client->ipu_mesh_manager().find(m_device_mesh_id);
  return mesh.device();
}

bool IpuPjRtExecutable::UseHostExecutable() const noexcept {
  // Always use host executable if present.
  return bool(m_host_executable);
}

PjRtExecutable* IpuPjRtExecutable::GetBaseExecutable() const {
  PjRtExecutable* base_executable =
      bool(m_host_executable)
          ? static_cast<PjRtExecutable*>(m_host_executable.get())
          : static_cast<PjRtExecutable*>(m_ipu_se_executable.get());
  return base_executable;
}

Status IpuPjRtExecutable::ValidateArgumentHandles(
    absl::Span<PjRtBuffer* const> argument_handles) const {
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const ComputationLayout& computation_layout =
      poplar_executable->module_config().entry_computation_layout();

  // Check argument number, shapes, and layouts.
  const int argument_shapes_size = argument_handles.size();
  if (argument_shapes_size != computation_layout.parameter_count()) {
    return InvalidArgument(
        "invalid number of arguments for computation: expected %d, got %u",
        computation_layout.parameter_count(), argument_shapes_size);
  }
  for (int i = 0, end = argument_handles.size(); i < end; ++i) {
    // TODO(b/187081154): Compare tiling info also.
    const auto& shape = argument_handles[i]->on_device_shape();
    if (!computation_layout.parameter_layout(i).MatchesLayoutInShape(
            shape, /*minor_to_major_only=*/false,
            /*ignore_fully_empty_tiling=*/true)) {
      LOG(WARNING) << absl::StrFormat(
          "Function '%s'. Argument does not match host shape or layout of "
          "computation "
          "parameter "
          "%d: want %s, got %s. This may impact performance.",
          poplar_executable->module().name(), i,
          ShapeUtil::HumanStringWithLayout(
              computation_layout.parameter_layout(i).shape()),
          ShapeUtil::HumanStringWithLayout(shape));
      // TODO: check outputs layout?
    }
  }
  return Status::OK();
}

Status IpuPjRtExecutable::MarkUnchangedDonatedBuffersAsSynchronizedOnSRAM(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles) const {
  const auto& inputs_aliasing = m_input_output_aliasing.inputs();
  for (std::size_t r = 0; r < argument_handles.size(); ++r) {
    const auto& replica_input_buffers = argument_handles[r];
    for (std::size_t idx = 0; idx < replica_input_buffers.size(); ++idx) {
      IpuPjRtBuffer* ipu_buffer =
          static_cast<IpuPjRtBuffer*>(replica_input_buffers[idx]);
      // Input donated buffer unchanged by the executable.
      if (inputs_aliasing[idx].type == IpuPjRtBufferDonationType::UNCHANGED) {
        ipu_buffer->ConvertToSRAMHostSynchronized();
      }
    }
  }
  return Status::OK();
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
IpuPjRtExecutable::ExecuteOnHost(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  TENSORFLOW_TRACEPOINT();
  // Only support single device for now.
  CHECK_EQ(argument_handles.size(), 1);
  CHECK(m_host_executable);

  const std::vector<PjRtBuffer*> in_ipu_buffers = argument_handles[0];
  // Extract host buffers.
  std::vector<PjRtBuffer*> in_host_buffers;
  in_host_buffers.reserve(in_ipu_buffers.size());
  for (std::size_t i = 0; i < in_ipu_buffers.size(); ++i) {
    auto* ipu_buffer = tensorflow::down_cast<IpuPjRtBuffer*>(in_ipu_buffers[i]);
    // Only supporting buffer on HOST at the moment.
    TF_ASSIGN_OR_RETURN(TfrtCpuBuffer * host_buffer,
                        ipu_buffer->GetHostBuffer());
    // Make sure there is a device assigned for successful execution.
    CHECK_NOTNULL(host_buffer->device());
    in_host_buffers.push_back(host_buffer);
  }
  // HOST execute.
  TF_ASSIGN_OR_RETURN(
      auto res_host_buffers,
      m_host_executable->Execute({in_host_buffers}, options, returned_futures));
  // Single host device (for now!)
  const std::size_t replica = 0;
  auto& replica_host_buffers = res_host_buffers[replica];

  // Wrap result buffers as IPU buffers.
  std::vector<std::unique_ptr<PjRtBuffer>> res_ipu_buffers;
  res_ipu_buffers.reserve(replica_host_buffers.size());
  for (std::size_t i = 0; i < replica_host_buffers.size(); ++i) {
    // No need to reference executable when run on HOST directly.
    auto res_buffer = IpuPjRtBuffer::CreateIpuBuffer(
        unique_down_cast<TfrtCpuBuffer>(std::move(replica_host_buffers[i])),
        IpuPjRtBufferLocation::HOST, m_devices[0], nullptr);
    res_ipu_buffers.push_back(std::move(res_buffer));
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
  result.push_back(std::move(res_ipu_buffers));
  return result;
}

Status IpuPjRtExecutable::CopyDeviceToHostBuffers(
    IpuPjRtRunOutputsRef* run_outputs_ref) {
  TENSORFLOW_TRACEPOINT();
  CHECK_NOTNULL(run_outputs_ref);
  // TODO: wait until all outputs are done.
  if (run_outputs_ref->IsAnyOnDeviceExpired()) {
    return InvalidArgument(
        "IPU on-device SRAM buffers are expired. Can not copy content back to "
        "host.");
  }
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();

  // Block until the buffers are ready (or in error state).
  auto& all_outputs = run_outputs_ref->output_buffers;
  const std::size_t num_replicas = all_outputs.size();
  const auto& output_infos = io_aliasing_map.GetEntryOutputInfos();
  for (const auto& replica_outputs : all_outputs) {
    for (const auto& buffer_and_status : replica_outputs) {
      IpuPjRtBuffer* output = buffer_and_status.buffer;
      // Wait until the buffer is ready (i.e. executable run done).
      // NOTE: blocking the main host thread (i.e. Python thread).
      TF_RETURN_IF_ERROR(output->BlockHostUntilReady());
    }
  }

  // Single use of Poplar::Engine at a time!
  std::scoped_lock l(m_poplar_engine_mutex);
  // TODO: check if not execute on HOST?
  poplar::Engine* engine = poplar_executable->Engine();
  CHECK_NOTNULL(engine);

  // Connect all on-device SRAM buffers to engine streams.
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    auto& replica_outputs = all_outputs[replica];
    for (std::size_t idx = 0; idx < replica_outputs.size(); ++idx) {
      IpuPjRtBuffer* output = replica_outputs[idx].buffer;
      const auto outinfo = output_infos[idx];
      if (output->location() == IpuPjRtBufferLocation::SRAM) {
        // Extract raw host buffer.
        const auto& outname = outinfo.Handles()[0];
        TF_ASSIGN_OR_RETURN(auto host_buffer_hold,
                            output->GetHostBufferWithHold(
                                TfrtCpuBuffer::ScopedHold::kUsage, true));
        const auto& raw_host_buffer = host_buffer_hold.buffer()->Buffers()[0];
        // Connect replica D2H stream to proper host buffer.
        engine->connectStreamToCallback(
            outname, replica,
            std::make_unique<IpuOutputStreamCallback>(raw_host_buffer));
      }
    }
  }
  LOG(INFO) << "Transfer D2H on-device SRAM buffers from IPU to HOST:"
            << name();
  // Host blocking call, transfering from SRAM to HOST.
  engine->run(PoplarProgramType::DEVICE_TO_HOST);

  // Mark all SRAM buffers as host synchronized.
  for (const auto& replica_outputs : all_outputs) {
    for (const auto& buffer_and_status : replica_outputs) {
      IpuPjRtBuffer* output = buffer_and_status.buffer;
      if (output->location() == IpuPjRtBufferLocation::SRAM) {
        output->MarkHostBufferSynchronized();
      }
    }
  }
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
