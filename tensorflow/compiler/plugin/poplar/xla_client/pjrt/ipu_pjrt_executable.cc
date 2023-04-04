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

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

namespace {

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

StatusOr<IpuPjRtBufferLocation> CheckInputDonatedBuffers(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const InputOutputAliasingMap& io_aliasing_map) {
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
  // All buffers already on SRAM => all good!
  if (all_buffers_on_sram) {
    return IpuPjRtBufferLocation::SRAM;
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

}  // namespace

/**
 * @brief IPU input stream callback.
 */
class IpuInputStreamCallback : public poplar::StreamCallback {
 public:
  IpuInputStreamCallback(std::shared_ptr<MaybeOwningCpuMemory> host_buffer)
      : m_host_buffer{std::move(host_buffer)} {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Not yet supported.
    return poplar::StreamCallback::Result::NotAvailable;
  }
  void fetch(void* dst) noexcept override {
    // Copy from input host TFRT buffer.
    std::memcpy(dst, m_host_buffer->data(), m_host_buffer->size());
  }
  void complete() noexcept override {}

 private:
  /** Input host buffer to stream to device. */
  std::shared_ptr<MaybeOwningCpuMemory> m_host_buffer;
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
  poplar_build_options.set_device_ordinal(ipu_mesh.id());
  DeviceAssignment poplar_device_assign(
      poplar_build_options.device_assignment());
  poplar_device_assign.Fill(ipu_mesh.id());
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
  // Other unsupported features.
  CHECK_EQ(poplar_executable->GetStreamInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetInfeedInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetOutfeedInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetSendInfos().size(), 0);
  CHECK_EQ(poplar_executable->GetRecvInfos().size(), 0);

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
    const std::vector<xla::PjRtBuffer*>& inbuffers) {
  IpuPjRtRunReplicaInputs inputs;
  // Highly inspired by TFRT CPU client/executable handling of buffers.
  const auto num_inputs = inbuffers.size();
  inputs.host_buffers.reserve(num_inputs);
  inputs.host_deps.reserve(num_inputs);
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
    }
    // Mark on-device input buffer as expired (i.e. input buffer donation).
    ipu_buffer->status()->MarkOnDeviceExpired();
    // Not supporting multiple buffers for now.
    CHECK_EQ(host_buffer_hold->Buffers().size(), 1);
    inputs.host_buffers.emplace_back(std::move(host_buffer_hold));
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
      engine->connectStreamToCallback(
          inname, replica,
          std::make_unique<IpuInputStreamCallback>(
              host_buffers[i].buffer()->Buffers()[0]));
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
      const auto& inbuffer = host_buffers[i].buffer()->Buffers()[0];
      engine->connectStreamToCallback(
          inname, replica, std::make_unique<IpuInputStreamCallback>(inbuffer));
    }
  }
}

StatusOr<IpuPjRtRunReplicaOutputs>
IpuPjRtRunReplicaOutputs::AllocateFromOutputInfos(
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos) {
  IpuPjRtRunReplicaOutputs outputs;
  // TODO: support tuple outputs?
  // TODO: improve alllocation to deal with memory fragmentation?
  const std::size_t num_outputs = output_infos.size();
  outputs.raw_host_buffers.reserve(num_outputs);
  for (const auto& output_info : output_infos) {
    TF_ASSIGN_OR_RETURN(auto out_buffer,
                        CreateRawHostBuffer(output_info.Shape()));
    outputs.raw_host_buffers.push_back(out_buffer);
    outputs.shapes.push_back(output_info.Shape());
  }
  return outputs;
}

std::vector<std::unique_ptr<PjRtBuffer>>
IpuPjRtRunReplicaOutputs::CreateIpuPjRtBuffers(
    const tfrt::AsyncValueRef<CpuEvent>& execute_event,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    PjRtDevice* ipu_device, IpuPjRtExecutable* executable) const {
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  // Create wrapping PjRt output buffers.
  const auto num_outputs = this->size();
  CHECK_EQ(num_outputs, output_infos.size());
  buffers.reserve(num_outputs);
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> sub_buffer;
    sub_buffer.push_back(raw_host_buffers[idx]);
    // Program execution writes to output buffers so it's a definition event.
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
    definition_events.push_back(execute_event.CopyRef());
    auto out_tracked_device_buffer =
        std::make_shared<TrackedTfrtCpuDeviceBuffer>(
            /*is_tuple=*/false, std::move(sub_buffer),
            std::move(definition_events));
    // IPU buffer location from the IO aliasing map information.
    const auto& outinfo = output_infos[idx];
    const auto location = IpuBufferLocationFromIOType(outinfo.GetType());
    // Wrapping everything into an IPU buffer!
    auto out_buffer = IpuPjRtBuffer::CreateIpuBuffer(
        this->shapes[idx], std::move(out_tracked_device_buffer), location,
        ipu_device, executable);
    buffers.push_back(std::move(out_buffer));
  }
  return buffers;
}

void IpuPjRtRunReplicaOutputs::ConnectStreamCallbacks(
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    int replica, poplar::Engine* engine) {
  const auto num_outputs = output_infos.size();
  for (std::size_t i = 0; i < num_outputs; ++i) {
    const auto& outinfo = output_infos[i];
    // Connect only streamed outputs.
    if (outinfo.IsStreaming()) {
      const auto& outname = outinfo.Handles()[0];
      engine->connectStreamToCallback(
          outname, replica,
          std::make_unique<IpuOutputStreamCallback>(raw_host_buffers[i]));
    }
  }
}

StatusOr<IpuPjRtRunState> IpuPjRtRunState::CreateWithIOBuffers(
    absl::Span<const std::vector<PjRtBuffer*>> all_input_handles,
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos) {
  const auto num_replicas = all_input_handles.size();
  IpuPjRtRunState run_state;
  run_state.all_inputs.reserve(num_replicas);
  run_state.all_outputs.reserve(num_replicas);
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    // Input buffers for the replica.
    TF_ASSIGN_OR_RETURN(auto inputs,
                        IpuPjRtRunReplicaInputs::CreateFromIpuPjRtBuffers(
                            all_input_handles[replica]));
    run_state.all_inputs.push_back(std::move(inputs));
    // Raw output buffers for the replica.
    TF_ASSIGN_OR_RETURN(
        auto outputs,
        IpuPjRtRunReplicaOutputs::AllocateFromOutputInfos(output_infos));
    run_state.all_outputs.push_back(std::move(outputs));
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
  // Random seed as well!
  engine->connectStream("__seed_stream", (void*)(&random_seed));
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
    const std::vector<InputOutputAliasingMap::OutputInfo>& output_infos,
    std::vector<PjRtDevice*> ipu_devices, IpuPjRtExecutable* executable) const {
  const auto num_replicas = this->num_replicas();
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> all_pjrt_outputs;
  all_pjrt_outputs.reserve(num_replicas);
  // Create IPU PjRt buffers wrapping raw outputs.
  for (std::size_t idx = 0; idx < num_replicas; ++idx) {
    all_pjrt_outputs.push_back(all_outputs[idx].CreateIpuPjRtBuffers(
        execute_event, output_infos, ipu_devices[idx], executable));
  }
  // Common run reference to all outputs, for handling on-device buffers.
  // TODO: handle returned status?
  IpuPjRtRunOutputsRef::CreateAndAssign(all_pjrt_outputs, executable);
  return all_pjrt_outputs;
}

StatusOr<std::shared_ptr<IpuPjRtRunOutputsRef>>
IpuPjRtRunOutputsRef::CreateAndAssign(
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>&
        all_out_buffers,
    IpuPjRtExecutable* executable) {
  CHECK_NOTNULL(executable);
  auto outref = std::make_shared<IpuPjRtRunOutputsRef>();
  outref->executable = executable;
  outref->output_buffers.resize(all_out_buffers.size());
  // Extract buffer pointer and status shared ptr.
  for (size_t r = 0; r < all_out_buffers.size(); ++r) {
    const auto& out_buffers = all_out_buffers[r];
    outref->output_buffers[r].reserve(out_buffers.size());
    for (const auto& buffer : out_buffers) {
      IpuPjRtBuffer* ipu_buffer =
          tensorflow::down_cast<IpuPjRtBuffer*>(buffer.get());
      CHECK_NOTNULL(ipu_buffer);
      outref->output_buffers[r].push_back(
          BufferAndStatusPtrs{ipu_buffer, ipu_buffer->status()});
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

void IpuPjRtRunOutputsRef::MarkOnDeviceExpired() {
  for (auto& outbuffers_ref : output_buffers) {
    for (auto& outbuf_ref : outbuffers_ref) {
      outbuf_ref.status->MarkOnDeviceExpired();
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

/////////////////////////////////////////////////////////////////////////////////////////
IpuPjRtExecutable::IpuPjRtExecutable(
    int64_t executable_id,
    std::unique_ptr<PjRtStreamExecutorExecutable> ipu_se_executable,
    std::unique_ptr<TfrtCpuExecutable> host_executable,
    const CompileOptions& compile_options, IpuPjRtClient* client)
    : m_executable_id{executable_id},
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
}
IpuPjRtExecutable::~IpuPjRtExecutable() {
  // Mark last run on-device expired, and nullify ptr.
  // Avoid error if trying to fetch on-device values.
  if (m_last_run_outputs_ref) {
    m_last_run_outputs_ref->MarkOnDeviceExpired();
    m_last_run_outputs_ref->executable = nullptr;
  }
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
  // Forward call directly to host executable.
  if (UseHostExecutable()) {
    return ExecuteOnHost(argument_handles, options, returned_futures);
  }

  auto* host_context = m_client->cpu_client()->GetHostContext();
  // Poplar executable and associated in/out mapping.
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();
  poplar::Engine* engine = poplar_executable->Engine();

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
  // Check streaming input buffers.
  TF_RETURN_IF_ERROR(
      CheckInputStreamingBuffers(argument_handles, io_aliasing_map));
  // Check on device/donated input buffers, and get the location of these input
  // buffers. On HOST location requires the buffers to be first streamed before
  // calling the main program.
  TF_ASSIGN_OR_RETURN(
      const auto inputs_donated_location,
      CheckInputDonatedBuffers(argument_handles, io_aliasing_map));

  LOG(INFO) << "Executing IPU computation " << name()
            << "; num_replicas=" << num_replicas()
            << " num_partitions=" << num_partitions()
            << "; num_addressable_devices=" << num_addressable_devices
            << "; num_inputs=" << num_inputs << " num_outputs=" << num_outputs;

  // PjRt run state with all replicas IO buffers.
  TF_ASSIGN_OR_RETURN(
      auto run_state,
      IpuPjRtRunState::CreateWithIOBuffers(
          argument_handles, io_aliasing_map.GetEntryOutputInfos()));

  // execute_event indicates whether ipu computation is complete and whether
  // there was an error.
  tfrt::AsyncValueRef<CpuEvent> execute_event;
  // Synchronously call generated function.
  execute_event = GetOrCreateReadyEvent(host_context);

  // Wrapping execute even as PjRt future status.
  std::optional<PjRtFuture<Status>> future;
  const bool fill_future = true;
  if (fill_future) {
    auto done_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
    execute_event.AndThen(
        [done_event = done_event.CopyRef(), event = execute_event.CopyRef()]() {
          Status s;
          if (auto* error = event.GetErrorIfPresent()) {
            s = InternalError("Compute error: %s", error->message);
          }
          done_event.emplace(std::move(s));
        });
    future = PjRtFuture<Status>(std::move(done_event));
  }
  // Returned outputs, for all replica, with IPU executable reference.
  auto outputs = run_state.CreateOutputIpuPjRtBuffers(
      execute_event, io_aliasing_map.GetEntryOutputInfos(), m_devices, this);
  // Returned futures, for all replica.
  if (returned_futures) {
    returned_futures.value().clear();
    for (std::size_t idx = 0; idx < run_state.num_replicas(); ++idx) {
      returned_futures.value().push_back(future.value());
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
  CHECK_NOTNULL(engine);

  // (Blocking) update of client state.
  // Use as coordination mechanism when required to load or reorganize IPUs.
  const auto [run_info, prev_client_state, new_client_state] =
      m_client->UpdateClientState(m_device_mesh_id, m_executable_id,
                                  m_last_run_outputs_ref);

  /////////////// RUNNING POPLAR ENGINE ///////////////
  /////////////// RUNNING POPLAR ENGINE ///////////////
  /////////////// RUNNING POPLAR ENGINE ///////////////

  // Should load on device => was mesh id & executable id was part of the
  // previous state.
  const auto prev_mesh = prev_client_state.FindByMeshId(m_device_mesh_id);
  if (prev_mesh == nullptr) {
    // Make sure the proper IPU Poplar mesh is attached.
    // Forcing detaching any other mesh overlapping. TODO: proper wait?
    m_client->ipu_mesh_manager().attach(m_device_mesh_id, true);
  }
  const bool load_poplar_engine =
      (prev_mesh == nullptr) || (prev_mesh->executable_id != m_executable_id);
  if (load_poplar_engine) {
    const auto& device_mesh =
        m_client->ipu_mesh_manager().find(m_device_mesh_id);
    LOG(INFO) << "Load IPU poplar engine " << name()
              << ", executable id: " << run_info.executable_id
              << ", on Poplar device with ID: " << m_device_mesh_id;
    // First call of the executable => load engine.
    engine->load(device_mesh.device());
  }

  // Wait for all inputs to be ready?
  for (std::size_t idx = 0; idx < run_state.num_replicas(); ++idx) {
    tfrt::Await(run_state.all_inputs[idx].host_deps);
  }

  // Transfer donated input buffers to IPU.
  if (inputs_donated_location == IpuPjRtBufferLocation::HOST) {
    LOG(INFO) << "Transfer buffers from HOST to IPU:" << name();
    // Connect on-device buffers H2D streams.
    run_state.ConnectH2DStreamDonatedBuffers(
        io_aliasing_map.GetEntryInputInfos(), engine);
    engine->run(PoplarProgramType::HOST_TO_DEVICE);
  }
  // Connect all replicas streams for running main program.
  run_state.ConnectStreamCallbacks(io_aliasing_map.GetEntryInputInfos(),
                                   io_aliasing_map.GetEntryOutputInfos(),
                                   engine);
  // Synchronous call => blocking thread!
  LOG(INFO) << "Run IPU poplar engine " << name()
            << "; executable id: " << run_info.executable_id
            << "; run id: " << run_info.run_id;
  engine->run(PoplarProgramType::MAIN_SEQUENCE);
  return outputs;
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
  throw std::runtime_error("Not implemented `Delete` on IPU.");
}

// True if on-device resources associated with the executable are freed.
bool IpuPjRtExecutable::IsDeleted() {
  throw std::runtime_error("Not implemented `IsDeleted` on IPU.");
}

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
      return InvalidArgument(
          "Argument does not match host shape or layout of computation "
          "parameter "
          "%d: want %s, got %s",
          i,
          ShapeUtil::HumanStringWithLayout(
              computation_layout.parameter_layout(i).shape()),
          ShapeUtil::HumanStringWithLayout(shape));
    }
  }
  return Status::OK();
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
IpuPjRtExecutable::ExecuteOnHost(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
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
        std::move(replica_host_buffers[i]), IpuPjRtBufferLocation::HOST,
        m_devices[0], nullptr);
    res_ipu_buffers.push_back(std::move(res_buffer));
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
  result.push_back(std::move(res_ipu_buffers));
  return result;
}

Status IpuPjRtExecutable::CopyDeviceToHostBuffers(
    IpuPjRtRunOutputsRef* run_outputs_ref) {
  // TODO: executable run mutex.
  // TODO: wait until all outputs are done.
  if (run_outputs_ref->IsAnyOnDeviceExpired()) {
    return InvalidArgument(
        "IPU on-device SRAM buffers are expired. Can not copy content back to "
        "host.");
  }
  LOG(INFO) << "Transfer on-device SRAM buffers from IPU to HOST:" << name();
  // TODO: check if not execute on HOST?
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_ipu_se_executable.get());
  const auto& io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();
  poplar::Engine* engine = poplar_executable->Engine();
  CHECK_NOTNULL(engine);

  const auto& output_infos = io_aliasing_map.GetEntryOutputInfos();
  auto& all_outputs = run_outputs_ref->output_buffers;
  const std::size_t num_replicas = all_outputs.size();
  // Connect all on-device SRAM buffers.
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    auto& replica_outputs = all_outputs[replica];
    for (std::size_t idx = 0; idx < replica_outputs.size(); ++idx) {
      IpuPjRtBuffer* output = replica_outputs[idx].buffer;
      const auto outinfo = output_infos[idx];
      if (output->location() == IpuPjRtBufferLocation::SRAM) {
        const auto& outname = outinfo.Handles()[0];
        // Extract raw host buffer.
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
  // Host blocking call.
  engine->run(PoplarProgramType::DEVICE_TO_HOST);
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
