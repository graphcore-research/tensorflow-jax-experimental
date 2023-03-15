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

Status CheckPoplarExecutableValid(PoplarExecutable* poplar_executable) {
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
  return Status::OK();
}

IpuPjRtExecutable::IpuPjRtExecutable(
    std::unique_ptr<PjRtStreamExecutorExecutable> se_executable,
    IpuPjRtClient* client)
    : m_se_executable{std::move(se_executable)}, m_client{client} {
  const auto& all_devices = client->addressable_devices();
  std::vector<int> device_ids(device_assignment().begin(),
                              device_assignment().end());
  // Build the collection of addressable devices for the executable.
  m_devices.clear();
  for (const int& id : device_ids) {
    auto it = std::find_if(all_devices.begin(), all_devices.end(),
                           [id](auto v) { return v->id() == id; });
    m_devices.push_back(*it);
  }
  // IPU device mesh associated.
  m_device_mesh_id =
      m_client->ipu_mesh_manager().find(device_assignment()).id();
  // A few checks!
  CHECK_GT(m_devices.size(), 0);
  CHECK_EQ(m_se_executable->executables().size(), 1);
}

PjRtClient* IpuPjRtExecutable::client() const { return m_client; }
// Unique name for this executable, e.g., HloModule name.
absl::string_view IpuPjRtExecutable::name() const {
  return m_se_executable->name();
}
int IpuPjRtExecutable::num_replicas() const {
  return m_se_executable->num_replicas();
}
int IpuPjRtExecutable::num_partitions() const {
  return m_se_executable->num_partitions();
}
int64_t IpuPjRtExecutable::SizeOfGeneratedCodeInBytes() const {
  throw std::runtime_error(
      "Not implemented `SizeOfGeneratedCodeInBytes` on IPU.");
}

const DeviceAssignment& IpuPjRtExecutable::device_assignment() const {
  return m_se_executable->device_assignment();
}
absl::Span<const PjRtExecutable::LogicalDeviceIds>
IpuPjRtExecutable::addressable_device_logical_ids() const {
  return m_se_executable->addressable_device_logical_ids();
}
absl::Span<PjRtDevice* const> IpuPjRtExecutable::addressable_devices() const {
  return m_devices;
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
IpuPjRtExecutable::GetHloModules() const {
  return m_se_executable->GetHloModules();
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
  auto* host_context = m_client->cpu_client()->GetHostContext();
  // Poplar executable, in/out map.
  PoplarExecutable* poplar_executable =
      GetPoplarExecutable(m_se_executable.get());
  poplar::Engine* engine = poplar_executable->Engine();
  const auto io_aliasing_map = poplar_executable->GetInputOutputAliasingMap();

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
  }
  LOG(INFO) << "Executing IPU computation " << name()
            << "; num_replicas=" << num_replicas()
            << " num_partitions=" << num_partitions()
            << " num_addressable_devices=" << num_addressable_devices
            << "; num_inputs=" << num_inputs << "; num_outputs=" << num_outputs;

  // TODO: support multiple replicas.
  CHECK_EQ(argument_handles.size(), 1);
  const int replica = 0;
  PjRtDevice* ipu_device = m_devices[replica];

  // Highly inspired by TFRT CPU client/executable handling of buffers.
  absl::InlinedVector<TfrtCpuBuffer::ScopedHold, 4> host_input_buffers;
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> host_input_deps;
  host_input_buffers.reserve(num_inputs);
  host_input_deps.reserve(num_inputs);
  // Extract host input buffers (and input events).
  for (int i = 0; i < num_inputs; ++i) {
    auto* ipu_buffer =
        tensorflow::down_cast<IpuPjRtBuffer*>(argument_handles[replica][i]);
    // Not supporting buffer donation for now.
    TF_ASSIGN_OR_RETURN(
        TfrtCpuBuffer::ScopedHold host_buffer_hold,
        ipu_buffer->GetHostBufferWithHold(TfrtCpuBuffer::ScopedHold::kUsage));
    if (!host_buffer_hold.ok()) {
      return InvalidArgument(
          "Invalid buffer passed to Execute() as argument %d to replica %d: "
          "%s",
          i, replica, host_buffer_hold.status().ToString());
    }
    // Definition events are never modified after buffer construction.
    for (const auto& ev : host_buffer_hold->DefinitionEvents()) {
      if (!ev.IsAvailable()) {
        host_input_deps.push_back(ev.CopyRCRef());
      }
    }
    // Not supporting multiple buffers for now.
    CHECK_EQ(host_buffer_hold->Buffers().size(), 1);
    host_input_buffers.emplace_back(std::move(host_buffer_hold));
  }

  // Create returned raw HOST buffers.
  // TODO: handle tuple args and tuple results?
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> result_buffers;
  result_buffers.reserve(num_outputs);
  for (const auto& output : io_aliasing_map.GetEntryOutputInfos()) {
    TF_ASSIGN_OR_RETURN(auto out_buffer, CreateRawHostBuffer(output.Shape()));
    result_buffers.push_back(out_buffer);
  }

  // execute_event indicates whether ipu computation is complete and whether
  // there was an error.
  tfrt::AsyncValueRef<CpuEvent> execute_event;
  // Synchronously call generated function.
  execute_event = GetOrCreateReadyEvent(host_context);

  const auto& device_mesh = m_client->ipu_mesh_manager().find(m_device_mesh_id);
  LOG(INFO) << "Load IPU poplar engine " << name()
            << " on Poplar device with ID: " << m_device_mesh_id;
  engine->load(device_mesh.device());
  // Connect all streams. TODO: use connect stream to callback.
  const int64_t custom_seed = 42;

  tfrt::Await(host_input_deps);

  // TODO: remove seed stream from executable.
  engine->connectStream("__seed_stream", (void*)(&custom_seed));
  // Input streams.
  for (std::size_t i = 0; i < num_inputs; ++i) {
    const auto& ininfo = io_aliasing_map.GetEntryInputInfos()[i];
    const auto& inname = ininfo.Handles()[0];
    const auto& inbuffer = host_input_buffers[i].buffer()->Buffers()[0];
    engine->connectStream(inname, inbuffer->data());
  }
  // Output streams.
  for (std::size_t i = 0; i < num_outputs; ++i) {
    const auto& outinfo = io_aliasing_map.GetEntryOutputInfos()[i];
    const auto& outname = outinfo.Handles()[0];
    engine->connectStream(outname, result_buffers[i]->data());
  }

  // Synchronous call => blocking thread!
  LOG(INFO) << "Run IPU poplar engine " << name();
  engine->run(PoplarProgramType::MAIN_SEQUENCE);

  // Create wrapping PjRt output buffers.
  std::vector<std::unique_ptr<PjRtBuffer>> res;
  res.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const Shape& shape = io_aliasing_map.GetEntryOutputInfos()[i].Shape();
    absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> sub_buffer;
    sub_buffer.push_back(std::move(result_buffers[i]));
    // Program execution writes to output buffers so it's a definition event.
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
    definition_events.push_back(execute_event.CopyRef());
    auto out_tracked_device_buffer =
        std::make_shared<TrackedTfrtCpuDeviceBuffer>(
            /*is_tuple=*/false, std::move(sub_buffer),
            std::move(definition_events));
    auto out_buffer = IpuPjRtBuffer::CreateIpuBufferOnHost(
        shape, std::move(out_tracked_device_buffer), ipu_device);
    res.push_back(std::move(out_buffer));
  }

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
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs;
  outputs.push_back(std::move(res));
  // Returned futures, for all replica.
  if (returned_futures) {
    returned_futures.value().clear();
    returned_futures.value().push_back(future.value());
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

}  // namespace poplarplugin
}  // namespace xla
