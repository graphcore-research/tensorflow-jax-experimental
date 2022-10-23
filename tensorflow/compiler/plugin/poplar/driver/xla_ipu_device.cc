/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/ipu_devices.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/kernels/no_op.h"

namespace xp = ::xla::poplarplugin;

namespace tensorflow {

class IpuDevice : public XlaDevice {
 public:
  IpuDevice(const SessionOptions& options, const XlaDevice::Options& devopts)
      : XlaDevice(options, devopts) {
    UseAcceleratorDeviceInfo();

    IPUDevices::GetActiveDevices().Add(this);
  }

  virtual ~IpuDevice() { IPUDevices::GetActiveDevices().Remove(this); }
};

class XlaIpuDeviceFactory : public XlaGraphcoreDeviceFactory {
 public:
  XlaIpuDeviceFactory()
      : XlaGraphcoreDeviceFactory(DEVICE_XLA_IPU, DEVICE_IPU_XLA_JIT,
                                  PLATFORM_NAME) {}

  std::unique_ptr<Device> CreateFromOptions(
      const SessionOptions& options,
      const XlaDevice::Options& devopts) const override {
    return absl::make_unique<IpuDevice>(options, devopts);
  }
};

REGISTER_IPU_XLA_DEVICES(DEVICE_XLA_IPU, XlaIpuDeviceFactory)

}  // namespace tensorflow
