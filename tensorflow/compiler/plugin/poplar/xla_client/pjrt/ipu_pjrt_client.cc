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
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_client.h"

#include <poplar/Graph.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/ipu_backend/ipu_executor.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_buffer.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_device.h"
#include "tensorflow/compiler/plugin/poplar/xla_client/pjrt/ipu_pjrt_executable.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace poplarplugin {

constexpr char kIpuName[] = "ipu";
static const char kIpuPlatformName[] = "ipu";

IpuPjRtClientState IpuPjRtClientState::Initialize(
    const IpuDeviceMeshManager& ipu_mesh_manager) {
  auto state = IpuPjRtClientState();
  state.m_active_meshes.clear();
  // Add individual IPU devices initially.
  for (const auto& m : ipu_mesh_manager.meshes()) {
    if (m.size() == 1) {
      // Proper IPU mesh id, no executable/run id.
      state.m_active_meshes.push_back(IpuPjRtMeshState{m.id(), 0, 0});
    }
  }
  return state;
}

IpuPjRtClientState IpuPjRtClientState::Update(
    const IpuPjRtExecutableRunInfo& run_info,
    const IpuDeviceMeshManager& ipu_mesh_manager) const {
  // Reconstruct new IPU client state.
  IpuPjRtClientState state_updated;
  const auto mesh_id = run_info.mesh_id;
  const auto is_mesh_active = this->IsActiveMesh(mesh_id);
  if (is_mesh_active) {
    // Is the IPU mesh already in used? => simple case.
    for (const auto& m : m_active_meshes) {
      if (m.mesh_id == mesh_id) {
        // Update the mesh state -> executable & run_id.
        state_updated.m_active_meshes.push_back(IpuPjRtMeshState{
            run_info.mesh_id, run_info.executable_id, run_info.run_id});
      } else {
        // Keep same IPU mesh state (i.e. same executable & run id).
        state_updated.m_active_meshes.push_back(m);
      }
    }
  } else {
    // More complicated case => new IPU mesh, need to remove all overlapping
    // meshes and introduce new one.
    const auto overlapping_mesh_ids =
        ipu_mesh_manager.overlappingMeshIds(mesh_id);
    const auto overlapping_set_ids = std::set<IpuDeviceMeshManager::IdType>(
        overlapping_mesh_ids.begin(), overlapping_mesh_ids.end());

    for (const auto& m : m_active_meshes) {
      if (overlapping_set_ids.find(m.mesh_id) == overlapping_set_ids.end()) {
        // Not overlapping with new IPU mesh => keep it.
        state_updated.m_active_meshes.push_back(m);
      }
    }
    // Add the new IPU mesh in use.
    state_updated.m_active_meshes.push_back(IpuPjRtMeshState{
        run_info.mesh_id, run_info.executable_id, run_info.run_id});
    // Sort by id, for consistency.
    std::sort(
        state_updated.m_active_meshes.begin(),
        state_updated.m_active_meshes.end(),
        [](const auto& m1, const auto& m2) { return m1.mesh_id < m2.mesh_id; });
  }
  return state_updated;
}

bool IpuPjRtClientState::IsActiveMesh(int mesh_id) const {
  return FindByMeshId(mesh_id) != nullptr;
}

const IpuPjRtMeshState* IpuPjRtClientState::FindByMeshId(
    int mesh_id) const noexcept {
  const auto it =
      std::find_if(m_active_meshes.begin(), m_active_meshes.end(),
                   [mesh_id](const auto& m) { return m.mesh_id == mesh_id; });
  if (it == m_active_meshes.end()) {
    return nullptr;
  }
  return &(*it);
}

const IpuPjRtMeshState* IpuPjRtClientState::FindByExecutableId(
    int executable_id) const noexcept {
  const auto it = std::find_if(m_active_meshes.begin(), m_active_meshes.end(),
                               [executable_id](const auto& m) {
                                 return m.executable_id == executable_id;
                               });
  if (it == m_active_meshes.end()) {
    return nullptr;
  }
  return &(*it);
}

/**
 * @brief Build IPU SE options to pass to IPU XLA stream executor backend.
 *
 * This method is bridging to the original IPU XLA backend: building the proper
 * flags and options to pass to the former.
 */
IpuOptions MakeIpuStreamExecutorOptions(
    const IpuDeviceMeshManager& ipu_mesh_manager,
    const IpuPjRtOptions& ipu_options) {
  CHECK_GT(ipu_mesh_manager.size(), 0);
  const auto& meshes = ipu_mesh_manager.meshes();
  IpuOptions se_options;

  // Create all IPU devices.
  if (ipu_mesh_manager.type() == poplar::TargetType::IPU_MODEL) {
    for (const auto& m : meshes) {
      // XLA client requiring auto count for IPU model.
      se_options.add_device_config()->set_auto_count(1);
    }
    // IPU model options.
    const auto& m = ipu_mesh_manager.meshes()[0];
    se_options.mutable_ipu_model_config()->set_tiles_per_ipu(
        m.num_tiles_per_ipu());
    se_options.mutable_ipu_model_config()->set_ipu_model_version(m.version());
    // Use `compile_ipu_code` option?
  } else {
    for (const auto& m : meshes) {
      // IPU hardware: use proper Poplar id.
      se_options.add_device_config()->set_cfg_index(m.id());
    }
  }

  // Prefetch config + rearrange copies
  se_options.set_prefetch_data_streams(ipu_options.prefetch_data_streams);
  auto* speed_size_cfg = se_options.mutable_speed_size_config();
  speed_size_cfg->set_always_rearrange_copies_on_the_host(
      ipu_options.always_rearrange_copies_on_the_host);

  // IOTilesConfig (default as IO tiles not supported)
  se_options.set_num_io_tiles(0);
  se_options.set_place_ops_on_io_tiles(false);
  se_options.set_io_tile_available_memory_proportion(0.9);
  return se_options;
}

/**
 * @brief Check consistency/compatibility between IPU device mesh and Poplar
 * target.
 */
Status CheckIpuMeshPoplarTarget(const IpuDeviceMeshInfo& mesh,
                                const poplar::Target& target) {
  if (mesh.type() != target.getTargetType()) {
    return FailedPrecondition("Inconsistent IPU target type");
  }
  CHECK_EQ(mesh.target().getNumIPUs(), target.getNumIPUs());
  CHECK_EQ(mesh.target().getTilesPerIPU(), target.getTilesPerIPU());
  return Status::OK();
}

/**
 * @brief Build XLA local client, used by stream executor client.
 *
 * Setting up proper IPU Poplar flags if necessary (e.g. IPU model).
 */
StatusOr<LocalClient*> GetIpuMeshXlaClient(
    const IpuDeviceMeshManager& ipu_mesh_manager) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("IPU"));

  // TODO: some check in XLA IPU client?
  // if (platform->VisibleDeviceCount() <= 0) {
  //   return FailedPrecondition("No visible Graphcore IPU devices.");
  // }
  // Set Poplar global flags. WARNING: const_cast to mutate directly flags...
  PoplarXlaFlags& poplar_flags =
      const_cast<PoplarXlaFlags&>(PoplarXlaFlags::Get());
  if (ipu_mesh_manager.type() == poplar::TargetType::IPU_MODEL) {
    // IPU model properties.
    poplar_flags.use_ipu_model = true;
    poplar_flags.ipu_model_tiles =
        ipu_mesh_manager.meshes()[0].num_tiles_per_ipu();
  } else {
    // Always make sure we deactivate IPU model!
    poplar_flags.use_ipu_model = false;
  }

  LocalClientOptions options;
  options.set_platform(platform);
  const auto& ipu_meshes = ipu_mesh_manager.meshes();
  CHECK_GT(ipu_meshes.size(), 0);
  // Available IPU meshes (using id provided by Poplar).
  std::set<int> allowed_devices;
  for (const auto& m : ipu_meshes) {
    allowed_devices.insert(m.id());
  }
  options.set_allowed_devices(allowed_devices);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

/**
 * @brief Build LocalDeviceState for each IPU (valid) mesh.
 */
StatusOr<std::vector<std::unique_ptr<LocalDeviceState>>>
BuildLocalSEMeshDeviceStates(LocalClient* xla_mesh_client, bool asynchronous,
                             const IpuDeviceMeshManager& ipu_mesh_manager,
                             const IpuPjRtOptions& ipu_options) {
  std::vector<std::unique_ptr<LocalDeviceState>> local_mesh_devices;
  const auto& meshes = ipu_mesh_manager.meshes();
  CHECK_GT(meshes.size(), 0);
  // Should have consistent num devices between IPU meshes and XLA client.
  CHECK_EQ(meshes.size(), xla_mesh_client->device_count());
  CHECK_EQ(meshes.size(), xla_mesh_client->backend().device_count());

  // IPU stream executor options.
  const auto ipu_se_options =
      MakeIpuStreamExecutorOptions(ipu_mesh_manager, ipu_options);
  // Make sure all stream executors are first reset.
  for (size_t i = 0; i < meshes.size(); ++i) {
    se::StreamExecutor* executor =
        xla_mesh_client->backend().stream_executor(i).ValueOrDie();
    auto* e = static_cast<PoplarExecutor*>(executor->implementation());
    e->Reset();
  }
  // Configure IPU XLA stream executors.
  for (size_t i = 0; i < meshes.size(); ++i) {
    // Configure Poplar executor IPU (mesh) device.
    se::StreamExecutor* executor =
        xla_mesh_client->backend().stream_executor(i).ValueOrDie();
    auto* e = static_cast<IpuExecutor*>(executor->implementation());
    TF_RETURN_IF_ERROR(e->ConfigurePoplarDevice(ipu_se_options));
    // Detach IPU device: stream executor should not need to keep hand on
    // device (only requiring target for compilation).
    e->DetachDevice();
    // Check that the Poplar target is consistent with IPU mesh description.
    CHECK(e->HasPoplarTarget());
    TF_RETURN_IF_ERROR(CheckIpuMeshPoplarTarget(meshes[i].info(),
                                                e->GetOrCreatePoplarTarget()));

    local_mesh_devices.push_back(absl::make_unique<LocalDeviceState>(
        executor, xla_mesh_client, LocalDeviceState::kComputeSynchronized,
        asynchronous,
        /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(local_mesh_devices);
}

/**
 * @brief Build IPU local stream-executor devices.
 */
std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalSEMeshDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& local_device_state : local_device_states) {
    int device_ordinal = local_device_state->device_ordinal();
    const se::DeviceDescription& description =
        local_device_state->executor()->GetDeviceDescription();
    auto device = absl::make_unique<PjRtStreamExecutorDevice>(
        device_ordinal, std::move(local_device_state), description.name());
    devices.push_back(std::move(device));
  }
  return devices;
}

/**
 * @brief Create IPU stream-executor mesh client: mapping all Poplar
 * devices (i.e. all possible combination of IPUs).
 */
StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> MakeIpuSEMeshClient(
    bool asynchronous, const IpuDeviceMeshManager& ipu_mesh_manager,
    const IpuPjRtOptions& ipu_options) {
  TF_ASSIGN_OR_RETURN(LocalClient * mesh_xla_client,
                      GetIpuMeshXlaClient(ipu_mesh_manager));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalDeviceState>> local_mesh_device_states,
      BuildLocalSEMeshDeviceStates(mesh_xla_client, asynchronous,
                                   ipu_mesh_manager, ipu_options));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> mesh_devices;
  mesh_devices = BuildLocalSEMeshDevices(std::move(local_mesh_device_states));
  return std::make_unique<PjRtStreamExecutorClient>(
      kIpuName, mesh_xla_client, std::move(mesh_devices),
      /*node_id=*/0,
      /*allocator=*/nullptr,
      /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);
}

IpuPjRtClient::IpuPjRtClient(bool asynchronous, int process_id,
                             IpuDeviceMeshManager ipu_mesh_manager,
                             std::vector<IpuPjRtDevice> devices,
                             const IpuPjRtOptions& options)
    : m_asynchronous{asynchronous},
      m_process_index{process_id},
      m_ipu_mesh_manager{std::move(ipu_mesh_manager)},
      m_devices{std::move(devices)},
      m_options{options} {
  m_ptr_devices.reserve(m_devices.size());
  for (auto& c : m_devices) {
    // Set client pointer in all local IPU devices.
    c.SetClient(this);
    // Internal device pointer array.
    m_ptr_devices.push_back(&c);
  }
  // Tfrt CPU client, handling buffers on host side.
  m_cpu_client = GetTfrtCpuClient(asynchronous, 1).value();
  CHECK_EQ(m_cpu_client->addressable_devices().size(), 1);

  // IPU stream executor client, representing IPU meshes.
  m_se_mesh_client =
      MakeIpuSEMeshClient(true, m_ipu_mesh_manager, m_options).ValueOrDie();

  // TODO: client state + attach single IPUs.
}
IpuPjRtClient::~IpuPjRtClient() {
  // Delete CPU and IPU stream executor clients.
  m_cpu_client.reset();
  m_se_mesh_client.reset();

  // XLA destroy client => necessary for testing different device
  // configurations.
  // TODO: more flexible mechanisme to only delete "ipu" XLA client, not all.
  ClientLibrary::DestroyLocalInstances();
}

int IpuPjRtClient::process_index() const { return m_process_index; }
int IpuPjRtClient::device_count() const { return m_devices.size(); }
int IpuPjRtClient::addressable_device_count() const { return m_devices.size(); }

absl::Span<PjRtDevice* const> IpuPjRtClient::devices() const {
  return m_ptr_devices;
}
absl::Span<PjRtDevice* const> IpuPjRtClient::addressable_devices() const {
  return m_ptr_devices;
}

StatusOr<PjRtDevice*> IpuPjRtClient::LookupDevice(int device_id) const {
  for (auto* ptr_device : m_ptr_devices) {
    if (device_id == ptr_device->id()) {
      return ptr_device;
    }
  }
  return InvalidArgument("No matching IPU device found for `device_id`: %d",
                         device_id);
}
StatusOr<PjRtDevice*> IpuPjRtClient::LookupAddressableDevice(
    int local_hardware_id) const {
  for (auto* ptr_device : m_ptr_devices) {
    if (local_hardware_id == ptr_device->local_hardware_id()) {
      return ptr_device;
    }
  }
  return InvalidArgument(
      "No matching IPU device found for local_hardware_id %d",
      local_hardware_id);
}

PjRtPlatformId IpuPjRtClient::platform_id() const {
  static const PjRtPlatformId kIpuId = tensorflow::Fingerprint64(kIpuName);
  return kIpuId;
}
absl::string_view IpuPjRtClient::platform_name() const {
  return kIpuPlatformName;
}
absl::string_view IpuPjRtClient::platform_version() const {
  // Use poplar::packageHash?
  static const std::string platform_version =
      absl::StrFormat("%s_sdk%s", kIpuPlatformName, poplar::versionString());
  return platform_version;
}
PjRtRuntimeType IpuPjRtClient::runtime_type() const { return kStreamExecutor; }

StatusOr<DeviceAssignment> IpuPjRtClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions != 1) {
    return InvalidArgument(
        "Unsupported number of partitions %i for IPU PjRt client.",
        num_partitions);
  }
  // Is there any mesh with the proper number of devices?
  const std::size_t num_devices = num_replicas * num_partitions;
  const bool is_mesh_existing = m_ipu_mesh_manager.count(num_devices);
  if (!is_mesh_existing) {
    return InvalidArgument(
        "No IPU mesh available with %i devices (%i replicas, %i partitions).",
        num_devices, num_replicas, num_partitions);
  }
  // Get the default mesh for this number of devices.
  const auto& default_mesh = m_ipu_mesh_manager.defaultMesh(num_devices);
  const auto& ipu_ids = default_mesh.info().ipuIds();
  auto device_assignment = DeviceAssignment(num_replicas, num_partitions);
  std::copy(ipu_ids.begin(), ipu_ids.end(), device_assignment.begin());
  return device_assignment;
}

StatusOr<std::unique_ptr<HloCostAnalysis>> IpuPjRtClient::GetHloCostAnalysis() {
  // Re-direct to StreamExecutor backend analysis.
  return m_se_mesh_client->GetHloCostAnalysis();
}

StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  // Device assignment required => use a default one if none.
  if (!options.executable_build_options.has_device_assignment()) {
    const int num_replicas = options.executable_build_options.num_replicas();
    const int num_partitions =
        options.executable_build_options.num_partitions();
    TF_ASSIGN_OR_RETURN(
        const auto device_assignment,
        this->GetDefaultDeviceAssignment(num_replicas, num_partitions));
    options.executable_build_options.set_device_assignment(device_assignment);
  }
  const auto poplar_compile_options =
      CreatePoplarCompileOptions(options, m_ipu_mesh_manager);
  // Compilation using stream executor IPU mesh client.
  LOG(INFO) << "IPU PJRT client: compiling device Poplar executable.";
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtExecutable> pjrt_executable,
      m_se_mesh_client->Compile(computation, poplar_compile_options));
  // Cast back to SE executable.
  std::unique_ptr<PjRtStreamExecutorExecutable> pjrt_se_executable(
      static_cast<PjRtStreamExecutorExecutable*>(pjrt_executable.release()));
  // Only supporting single executable for now.
  CHECK_EQ(pjrt_se_executable->executables().size(), 1);
  // Perform a couple of checks on the underlying Poplar executable.
  auto poplar_executable = GetPoplarExecutable(pjrt_se_executable.get());
  TF_RETURN_IF_ERROR(CheckPoplarExecutableValid(poplar_executable, options));

  // Optional host executable, when no Poplar engine compiled.
  auto host_executable = std::unique_ptr<TfrtCpuExecutable>{nullptr};
  const bool host_executable_required =
      (poplar_executable->Engine() == nullptr);
  if (host_executable_required) {
    LOG(INFO) << "IPU PJRT client: compiling CPU/HOST executable.";
    // TODO: convert compile options to fit CPU client.
    TF_ASSIGN_OR_RETURN(auto executable,
                        m_cpu_client->Compile(computation, options));
    // Cast back to CPU TFRT executable.
    host_executable = std::unique_ptr<TfrtCpuExecutable>{
        static_cast<TfrtCpuExecutable*>(executable.release())};
  }

  // Build IPU PjRt executable, wrapping IPU and HOST executables.
  return std::unique_ptr<PjRtExecutable>(std::make_unique<IpuPjRtExecutable>(
      m_executable_id_counter.increment(), std::move(pjrt_se_executable),
      std::move(host_executable), options, this));
}
StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  // TODO: convert back to XLA.
  // FIXME: LLVM bug?
  return Unimplemented("Not implemented MLIR `Compile` on IPU.");
}

// Generates a unique fingerprint for `executable`, may be std::nullopt.
StatusOr<std::optional<std::string>> IpuPjRtClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
  // Same as stream executor implementation. TODO: more useful answer?
  return std::optional<std::string>();
}

StatusOr<std::string> IpuPjRtClient::SerializeExecutable(
    const PjRtExecutable& executable) const {
  return Unimplemented("Not implemented `SerializeExecutable` on IPU.");
}
// Deserializes a serialized executable as produced by
// SerializeExecutable(). `serialized` must have been produced by a client of
// the same platform and version as this one.
StatusOr<std::unique_ptr<PjRtExecutable>> IpuPjRtClient::DeserializeExecutable(
    absl::string_view serialized, CompileOptions options) {
  return Unimplemented("Not implemented `DeserializeExecutable` on IPU.");
}

// Creates a buffer on the device without initializing or copying any data.
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device) {
  return Unimplemented("Not implemented `CreateUninitializedBuffer` on IPU.");
}
StatusOr<std::unique_ptr<PjRtClient::AsyncBufferTransferManager>>
IpuPjRtClient::CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                             PjRtDevice* device) {
  return Unimplemented(
      "Not implemented `CreateBuffersForAsyncTransfer` on IPU.");
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  // Create IPU buffer on the HOST. Will be transfered on IPU when required.
  // Assigning default HOST device to the cpu buffer.
  PjRtDevice* host_device = m_cpu_client->addressable_devices()[0];
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> cpu_buffer,
                      m_cpu_client->BufferFromHostBuffer(
                          data, type, dims, byte_strides, host_buffer_semantics,
                          std::move(on_done_with_host_buffer), host_device));
  return IpuPjRtBuffer::CreateIpuBufferOnHost(std::move(cpu_buffer), device);
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtDevice* device) {
  return Unimplemented("Not implemented `BufferFromHostLiteral` on IPU.");
}
StatusOr<std::unique_ptr<PjRtBuffer>> IpuPjRtClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback) {
  return Unimplemented("Not implemented `CreateViewOfDeviceBuffer` on IPU.");
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtClient::MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                                           PjRtDevice* device,
                                           PjRtCrossHostRecvNotifier notifier) {
  // Not necessary on single process?
  return Unimplemented("Not implemented `MakeCrossHostReceiveBuffers` on IPU.");
}
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
IpuPjRtClient::MakeCrossHostReceiveBuffersForGather(
    absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
    PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) {
  // Not necessary on single process?
  return Unimplemented(
      "Not implemented `MakeCrossHostReceiveBuffersForGather` on IPU.");
}

StatusOr<ChannelHandle> IpuPjRtClient::CreateChannelHandle() {
  return Unimplemented("Not implemented `CreateChannelHandle` on IPU.");
}
StatusOr<ChannelHandle> IpuPjRtClient::CreateDeviceToHostChannelHandle() {
  return Unimplemented(
      "Not implemented `CreateDeviceToHostChannelHandle` on IPU.");
}
StatusOr<ChannelHandle> IpuPjRtClient::CreateHostToDeviceChannelHandle() {
  return Unimplemented(
      "Not implemented `CreateHostToDeviceChannelHandle` on IPU.");
}

Status IpuPjRtClient::Defragment() {
  return Unimplemented("Not implemented `Defragment` on IPU.");
}

const IpuDeviceMeshManager& IpuPjRtClient::ipu_mesh_manager() const noexcept {
  return m_ipu_mesh_manager;
}

std::tuple<IpuPjRtExecutableRunInfo, IpuPjRtClientState, IpuPjRtClientState>
IpuPjRtClient::UpdateClientState(int mesh_id, int executable_id) {
  // Make sure we can't have multiple updates in parallel.
  std::lock_guard<std::mutex> lk(m_client_state_mutex);
  const auto current_state = m_client_state;
  // TODO: pass PjRtFuture as well (or generate here?)
  const auto run_info =
      IpuPjRtExecutableRunInfo{mesh_id, executable_id, next_run_id()};
  m_client_state = m_client_state.Update(run_info, m_ipu_mesh_manager);
  return std::make_tuple(run_info, current_state, m_client_state);
}

// Factory methods.
IpuDeviceMeshManager CreateIpuDeviceMeshManager(
    const IpuPjRtOptions& ipu_options) {
  // Visible devices option not yet supported.
  CHECK(!ipu_options.visible_devices.has_value());
  if (ipu_options.use_ipu_model) {
    return IpuDeviceMeshManager::createIpuModelManager(
        ipu_options.ipu_model_num_tiles, ipu_options.ipu_model_version);
  }
  return IpuDeviceMeshManager::createIpuManager();
}

std::vector<IpuPjRtDevice> CreateIpuDevices(
    const IpuDeviceMeshManager& mesh_manager) {
  std::vector<IpuPjRtDevice> devices;
  for (const auto& m : mesh_manager.meshes()) {
    // Single IPU device mesh only.
    if (m.size() == 1) {
      devices.push_back(IpuPjRtDevice(m.info()));
    }
  }
  return devices;
}

StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(
    bool asynchronous, const IpuPjRtOptions& ipu_options) {
  // Default local process id?
  int process_id = 0;
  auto mesh_manager = CreateIpuDeviceMeshManager(ipu_options);
  auto devices = CreateIpuDevices(mesh_manager);

  return std::unique_ptr<PjRtClient>(std::make_unique<IpuPjRtClient>(
      asynchronous, process_id, std::move(mesh_manager), std::move(devices),
      ipu_options));
}

}  // namespace poplarplugin
}  // namespace xla
