/* Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

class IpuDevice : public PjRtStreamExecutorDevice {
 public:
  IpuDevice(int id, std::unique_ptr<LocalDeviceState> local_device_state,
            std::string device_kind);
};

struct IpuConfig {
  size_t num_ipus = 1;

  /* The data which is streamed to/from the device might be stored in different
  layouts on the device and on the host. If so, rearrangement is performed on
  the device by default. By enabling this option the rearrangement will be
  performed on the host at the expense of latency.*/
  bool always_rearrange_copies_on_the_host = true;

  /* If true (default), prefetching of data for data streams on the host will be
    overlapped with execution on the IPU.*/
  bool prefetch_data_streams = true;

  /* Number of tiles to reserve for I/O.*/
  int64 num_io_tiles = 0;

  /* Whether to place TensorFlow I/O operations on the I/O tiles.*/
  bool place_ops_on_io_tiles = false;

  /* Proportion of I/O tiles' memory which can be used to store data in, with the
  remaining memory assumed to be used by code. If the size of data which is to
  be stored on I/O tiles exceeds the total I/O tiles memory multiplied by this
  proportion, then a warning message will appear and the operations will not
  be placed on I/O tiles.*/
  double io_tile_available_memory_proportion = 0.9;
};

StatusOr<std::unique_ptr<PjRtClient>> GetIpuClient(
    bool asynchronous, const IpuConfig& config);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_XLA_CLIENT_PJRT_IPU_DEVICE_H_