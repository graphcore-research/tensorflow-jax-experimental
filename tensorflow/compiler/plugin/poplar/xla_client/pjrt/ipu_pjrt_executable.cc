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

/**
 * @brief IPU input stream callback.
 */
class IpuInputStreamCallback : public poplar::StreamCallback {
 public:
  IpuInputStreamCallback(std::size_t index, IpuPjRtRunReplicaInputs* inputs)
      : m_index{index}, m_inputs{inputs} {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Not yet supported.
    return poplar::StreamCallback::Result::NotAvailable;
  }
  void fetch(void* dst) noexcept override {
    // Too complicated => simplify?
    const auto& inbuffer =
        m_inputs->host_buffers[m_index].buffer()->Buffers()[0];
    // Copy from input host TFRT buffer.
    std::memcpy(dst, inbuffer->data(), inbuffer->size());
  }
  void complete() noexcept override {}

 private:
  /** Output index (in the following data structure). */
  std::size_t m_index;
  /** Pointer to raw host input buffers. */
  IpuPjRtRunReplicaInputs* m_inputs;
};

/**
 * @brief IPU output stream callback.
 */
class IpuOutputStreamCallback : public poplar::StreamCallback {
 public:
  IpuOutputStreamCallback(std::size_t index, IpuPjRtRunReplicaOutputs* outputs)
      : m_index{index}, m_outputs{outputs} {}

  poplar::StreamCallback::Result prefetch(void* dst) noexcept override {
    // Not supported for outputs.
    return poplar::StreamCallback::Result::NotAvailable;
  }
  void fetch(void* src) noexcept override {
    const auto& outbuffer = m_outputs->raw_host_buffers[m_index];
    // Copy to output host TFRT buffer.
    std::memcpy(outbuffer->data(), src, outbuffer->size());
  }
  void complete() noexcept override {}

 private:
  /** Output index (in the following data structure). */
  std::size_t m_index;
  /** Pointer to raw host output buffers. */
  IpuPjRtRunReplicaOutputs* m_outputs;
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
    if (!v.IsStreaming()) {
      return FailedPrecondition(
          "IPU PjRt client only supports streaming inputs: %s.",
          v.Handles()[0]);
    }
  }
  for (const auto& v : io_aliasing_map.GetEntryOutputInfos()) {
    CHECK_EQ(v.Handles().size(), 1);
    CHECK(!v.Shape().IsTuple());
    if (!v.IsStreaming()) {
      return FailedPrecondition(
          "IPU PjRt client only supports streaming outputs: %s.",
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
  CHECK_EQ(device_assignment.computation_count(), 1);
  CHECK_EQ(device_assignment.replica_count(),
           poplar_executable->GetReplicationFactor());
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
        ipu_buffer->GetHostBufferWithHold(TfrtCpuBuffer::ScopedHold::kUsage));
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
    // TODO: pass directly raw host buffer?
    engine->connectStreamToCallback(
        inname, replica, std::make_unique<IpuInputStreamCallback>(i, this));
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
    PjRtDevice* ipu_device) const {
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  // Create wrapping PjRt output buffers.
  const auto num_outputs = this->size();
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
    auto out_buffer = IpuPjRtBuffer::CreateIpuBufferOnHost(
        this->shapes[idx], std::move(out_tracked_device_buffer), ipu_device);
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
    const auto& outname = outinfo.Handles()[0];
    engine->connectStreamToCallback(
        outname, replica, std::make_unique<IpuOutputStreamCallback>(i, this));
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
  const auto num_replicas = all_inputs.size();
  // Connect streams from all replicas.
  for (std::size_t replica = 0; replica < num_replicas; ++replica) {
    all_inputs[replica].ConnectStreamCallbacks(input_infos, replica, engine);
    all_outputs[replica].ConnectStreamCallbacks(output_infos, replica, engine);
  }
  // Random seed as well!
  engine->connectStream("__seed_stream", (void*)(&random_seed));
}

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtRunState::CreateOutputIpuPjRtBuffers(
    const tfrt::AsyncValueRef<CpuEvent>& execute_event,
    std::vector<PjRtDevice*> ipu_devices) const {
  const auto num_replicas = all_inputs.size();
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> all_pjrt_outputs;
  all_pjrt_outputs.reserve(num_replicas);
  for (std::size_t idx = 0; idx < num_replicas; ++idx) {
    all_pjrt_outputs.push_back(
        all_outputs[idx].CreateIpuPjRtBuffers(execute_event, ipu_devices[idx]));
  }
  return all_pjrt_outputs;
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
  CHECK_EQ(m_ipu_se_executable->executables().size(), 1);
  CHECK(m_compile_options.executable_build_options.has_device_assignment());
  // Should have at least one or the other!
  CHECK(bool(m_ipu_se_executable) || bool(m_host_executable));
}

PjRtClient* IpuPjRtExecutable::client() const { return m_client; }
// Unique name for this executable, e.g., HloModule name.
absl::string_view IpuPjRtExecutable::name() const {
  return m_ipu_se_executable->name();
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
  return m_ipu_se_executable->GetHloModules();
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
  const auto io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();
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

  // Poplar engine: if null, means it should be executed on HOST directly
  // (constant, scalar, ...). NOTE: check engine after inputs status.
  CHECK_NOTNULL(engine);

  // execute_event indicates whether ipu computation is complete and whether
  // there was an error.
  tfrt::AsyncValueRef<CpuEvent> execute_event;
  // Synchronously call generated function.
  execute_event = GetOrCreateReadyEvent(host_context);

  // (Blocking) update of client state.
  // Use as coordination mechanism when required to load or reorganize IPUs.
  const auto [run_info, prev_client_state, new_client_state] =
      m_client->UpdateClientState(m_device_mesh_id, m_executable_id);

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
  // Connect all replicas streams.
  run_state.ConnectStreamCallbacks(io_aliasing_map.GetEntryInputInfos(),
                                   io_aliasing_map.GetEntryOutputInfos(),
                                   engine);

  // Synchronous call => blocking thread!
  LOG(INFO) << "Run IPU poplar engine " << name()
            << "; executable id: " << run_info.executable_id
            << "; run id: " << run_info.run_id;
  engine->run(PoplarProgramType::MAIN_SEQUENCE);

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
  // Result({/*future=*/std::move(future), /*buffers=*/std::move(res)});

  // Returned outputs, for all replica.
  auto outputs = run_state.CreateOutputIpuPjRtBuffers(execute_event, m_devices);
  // Returned futures, for all replica.
  if (returned_futures) {
    returned_futures.value().clear();
    for (std::size_t idx = 0; idx < run_state.num_replicas(); ++idx) {
      returned_futures.value().push_back(future.value());
    }
  }
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

  // Wrap result buffers as IPU buffers.
  std::vector<std::unique_ptr<PjRtBuffer>> res_ipu_buffers;
  res_ipu_buffers.reserve(res_host_buffers.size());
  for (std::size_t i = 0; i < res_host_buffers.size(); ++i) {
    auto res_buffer = IpuPjRtBuffer::CreateIpuBufferOnHost(
        std::move(res_host_buffers[0][i]), m_devices[0]);
    res_ipu_buffers.push_back(std::move(res_buffer));
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
  result.push_back(std::move(res_ipu_buffers));
  return result;
}

}  // namespace poplarplugin
}  // namespace xla
