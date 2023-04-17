# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""IPU PRJT device API unit tests (using IPU model)."""
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import gc
import time

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xla_extension
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import (
    IpuDeviceMeshManager, IpuPjRtOptions, get_ipu_client, IpuTargetType, IpuPjRtDevice,
    IpuPjRtOptions, make_ipu_client, parse_ipu_env_flags, make_ipu_legacy_config,
    make_ipu_pjrt_options
)

ops = xla_client.ops
xe = xla_client._xla

# Skipping some tests if no local IPU hardware.
num_ipu_hw_available = IpuDeviceMeshManager.num_ipu_hardware_available()
ipu_hw_available = num_ipu_hw_available > 0
# Use asynchronous backend?
async_backend = True


def approximate_timer(fn, N: int) -> np.ndarray:
  """Approximate/basic benchmark timer of a Python callable.
  """
  timings = []
  for _ in range(N):
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    timings.append(end - start)
  return np.array(timings)


class IpuPjrtClientFactoryTest(parameterized.TestCase):

  def testIpuPjRtclient__parse_ipu_env_flags__xla_jax_ipu_flags(self):
    env = {
        "XLA_IPU_PLATFORM_DEVICE_COUNT": "1",
        "XLA_IPU_PLATFORM_ALWAYS_REARRANGE_COPIES_ON_THE_HOST": "true",
        "JAX_IPU_USE_MODEL": "true",
        "JAX_IPU_MODEL_NUM_TILES": "8",
        "JAX_IPU_USE_LEGACY_CLIENT": "true",
        "XLA_IPU_PLATFORM_NUM_IO_TILES": "10",
        "PATH": "blalba"
    }
    flags = parse_ipu_env_flags(env)
    self.assertEqual(
        flags, {
            'always_rearrange_copies_on_the_host': 'true',
            'device_count': '1',
            'model_num_tiles': '8',
            'num_io_tiles': '10',
            'use_legacy_client': 'true',
            'use_model': 'true'
        }
    )

  def testIpuPjRtclient__make_ipu_legacy_config__from_flags(self):
    flags = {
        'always_rearrange_copies_on_the_host': 'true',
        'device_count': '2',
        'model_num_tiles': '8',
        'num_io_tiles': '10',
        'use_legacy_client': 'true',
        'use_model': 'true'
    }
    ipu_config = make_ipu_legacy_config(flags)
    self.assertTrue(ipu_config.always_rearrange_copies_on_the_host)
    self.assertFalse(ipu_config.prefetch_data_streams)
    self.assertFalse(ipu_config.place_ops_on_io_tiles)
    self.assertEqual(ipu_config.num_ipus, 2)
    self.assertEqual(ipu_config.num_io_tiles, 10)

  def testIpuPjRtclient__make_ipu_legacy_config__proper_default_device_count(self):
    flags = {
        'always_rearrange_copies_on_the_host': 'true',
        'device_count': '-1',
        'model_num_tiles': '8',
        'num_io_tiles': '10',
        'use_legacy_client': 'true',
        'use_model': 'true'
    }
    ipu_config = make_ipu_legacy_config(flags)
    self.assertEqual(ipu_config.num_ipus, 1)

  def testIpuPjRtclient__make_ipu_pjrt_options__from_flags(self):
    flags = {
        'always_rearrange_copies_on_the_host': 'true',
        'device_count': '3',
        'visible_devices': '0,2,3',
        'model_num_tiles': '8',
        'num_io_tiles': '10',
        'use_legacy_client': 'true',
        'use_model': 'true',
        'execute_on_host_flops_limit': '2.0',
    }
    ipu_options = make_ipu_pjrt_options(flags)
    self.assertIsInstance(ipu_options, IpuPjRtOptions)
    self.assertEqual(ipu_options.num_devices, 3)
    self.assertTrue(ipu_options.use_ipu_model)
    self.assertEqual(ipu_options.ipu_model_num_tiles, 8)
    self.assertTrue(ipu_options.always_rearrange_copies_on_the_host)
    self.assertEqual(ipu_options.visible_devices, {0, 2, 3})
    self.assertEqual(ipu_options.execute_on_host_flops_limit, 2)

  def testIpuPjRtclient__make_ipu_client__from_env_variables(self):
    env = {
        "XLA_IPU_PLATFORM_DEVICE_COUNT": "2",
        "XLA_IPU_PLATFORM_ALWAYS_REARRANGE_COPIES_ON_THE_HOST": "false",
        "JAX_IPU_USE_MODEL": "true",
        "JAX_IPU_MODEL_NUM_TILES": "16",
        "JAX_IPU_USE_LEGACY_CLIENT": "false",
        "XLA_IPU_PLATFORM_NUM_IO_TILES": "0",
        "PATH": "blalba"
    }
    client = make_ipu_client(env)
    self.assertIsInstance(client, xla_client.Client)
    self.assertEqual(client.local_device_count(), 2)
    self.assertEqual(client.device_count(), 2)
    for d in client.devices():
      self.assertIsInstance(d, IpuPjRtDevice)
      self.assertEqual(d.num_tiles, 16)

  def testIpuPjRtclient__make_ipu_client__ipu_model_default_num_devices(self):
    env = {
        "XLA_IPU_PLATFORM_DEVICE_COUNT": "-1",
        "XLA_IPU_PLATFORM_ALWAYS_REARRANGE_COPIES_ON_THE_HOST": "false",
        "JAX_IPU_USE_MODEL": "true",
        "JAX_IPU_MODEL_NUM_TILES": "16",
        "JAX_IPU_USE_LEGACY_CLIENT": "false",
        "XLA_IPU_PLATFORM_NUM_IO_TILES": "0",
        "PATH": "blalba"
    }
    client = make_ipu_client(env)
    self.assertIsInstance(client, xla_client.Client)
    self.assertEqual(client.local_device_count(), 1)
    self.assertEqual(client.device_count(), 1)


class IpuPjrtClientBaseTest(parameterized.TestCase):

  def setUp(self):
    super(IpuPjrtClientBaseTest, self).setUp()

  def testIpuPjRtclient__get_ipu_client__base_properties(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_client = get_ipu_client(asynchronous=async_backend, ipu_options=ipu_options)
    self.assertIsInstance(ipu_client, xla_extension.Client)
    self.assertEqual(ipu_client.process_index(), 0)
    self.assertEqual(ipu_client.host_id(), 0)
    self.assertEqual(ipu_client.platform, "ipu")
    self.assertEqual(ipu_client.platform_version, "ipu_sdk3.1.0 (e12d5f9f01)")
    self.assertEqual(ipu_client.runtime_type, "stream_executor")

  def testIpuPjRtclient__get_ipu_client__ipu_model_local_devices(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_client = get_ipu_client(asynchronous=async_backend, ipu_options=ipu_options)
    ipu_devices = ipu_client.local_devices()

    self.assertEqual(ipu_client.local_device_count(), 1)
    self.assertEqual(ipu_client.device_count(), 1)
    self.assertEqual(len(ipu_devices), 1)

    ipu_device = ipu_devices[0]
    self.assertIsInstance(ipu_device, IpuPjRtDevice)
    self.assertEqual(ipu_device.id, 0)
    self.assertEqual(ipu_device.num_tiles, 16)
    self.assertEqual(ipu_device.version, "ipu21")
    self.assertEqual(ipu_device.type, IpuTargetType.IPU_MODEL)

  @unittest.skipIf(num_ipu_hw_available < 2, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__ipu_hardware_local_devices(self):
    ipu_client = get_ipu_client(
        asynchronous=async_backend, ipu_options=IpuPjRtOptions(num_devices=2)
    )
    ipu_devices = ipu_client.local_devices()

    self.assertEqual(len(ipu_devices), 2)
    self.assertEqual([d.id for d in ipu_devices], list(range(len(ipu_devices))))
    self.assertEqual({d.num_tiles for d in ipu_devices}, {1472})
    self.assertEqual({d.version for d in ipu_devices}, {"ipu2"})
    self.assertEqual({d.type for d in ipu_devices}, {IpuTargetType.IPU})

  @unittest.skipIf(num_ipu_hw_available < 4, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__visible_devices_only(self):
    ipu_client = get_ipu_client(
        asynchronous=async_backend,
        ipu_options=IpuPjRtOptions(visible_devices={0, 2, 3})
    )
    ipu_devices = ipu_client.local_devices()

    self.assertEqual(len(ipu_devices), 3)
    self.assertEqual([d.id for d in ipu_devices], list(range(len(ipu_devices))))
    self.assertEqual([d.local_hardware_id for d in ipu_devices], [0, 2, 3])
    self.assertEqual({d.num_tiles for d in ipu_devices}, {1472})
    self.assertEqual({d.version for d in ipu_devices}, {"ipu2"})
    self.assertEqual({d.type for d in ipu_devices}, {IpuTargetType.IPU})

  @unittest.skipIf(num_ipu_hw_available < 2, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__multi_clients_create_delete(self):
    ipu_client = get_ipu_client(
        asynchronous=async_backend, ipu_options=IpuPjRtOptions(num_devices=2)
    )
    ipu_devices = ipu_client.local_devices()
    self.assertEqual([d.local_hardware_id for d in ipu_devices], [0, 1])
    # Check resources are properly freed.
    del ipu_devices
    del ipu_client
    gc.collect()
    # New IPU client should attach the same IPUs.
    ipu_client = get_ipu_client(
        asynchronous=async_backend, ipu_options=IpuPjRtOptions(num_devices=2)
    )
    ipu_devices = ipu_client.local_devices()
    self.assertEqual([d.local_hardware_id for d in ipu_devices], [0, 1])


class IpuPjrtClientBufferTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # IPU model with 1 device.
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=4, ipu_model_version="ipu2"
    )
    cls.backend = get_ipu_client(asynchronous=async_backend, ipu_options=ipu_options)

  @classmethod
  def tearDownClass(cls):
    # Force to delete the IPU client.
    del cls.backend
    gc.collect()

  def setUp(self):
    super(IpuPjrtClientBufferTest, self).setUp()

  def testIpuPjRtclient__buffer_from_pyval__base_properties(self):
    pyval = np.array([[1., 2.]], np.float32)
    local_buffer = self.backend.buffer_from_pyval(pyval)
    xla_shape = local_buffer.xla_shape()
    self.assertEqual(xla_shape.dimensions(), (1, 2))
    self.assertEqual(np.dtype(xla_shape.element_type()), np.dtype(np.float32))
    self.assertEqual(local_buffer.device(), self.backend.devices()[0])

  def testIpuPjRtclient__buffer__base_signatures(self):
    # When extending `DeviceArrayBase`, the object behaves as a `DeviceArray`
    # and thus needs to correctly implement the following methods.
    arg = np.array([[1., 2., 3.]], np.float32)
    buffer = self.backend.buffer_from_pyval(arg)
    if not isinstance(buffer, xla_client.DeviceArrayBase):
      raise unittest.SkipTest(
          "The objectof type {} do not extend DeviceArrayBase".format(type(buffer))
      )

    self.assertEqual(buffer.__array_priority__, 100)
    self.assertEqual(buffer.shape, (1, 3))
    self.assertEqual(buffer.dtype, np.float32)
    self.assertEqual(buffer.size, 3)
    self.assertEqual(buffer.ndim, 2)

    self.assertIs(buffer, buffer.block_until_ready())
    self.assertTrue(buffer.is_ready())
    buffer.delete()
    with self.assertRaises(xla_client.XlaRuntimeError):
      buffer.block_until_ready()
    with self.assertRaises(xla_client.XlaRuntimeError):
      buffer.is_ready()

  def testIpuPjRtclient__buffer__copy_to_host(self):
    arg0 = np.array([[1., 2.]], np.float32)
    arg1 = np.array([[3., 4.]], np.float32)
    arg0_buffer = self.backend.buffer_from_pyval(arg0)
    arg1_buffer = self.backend.buffer_from_pyval(arg1)
    # Prefetch two buffers using copy_to_host_async, and then retrieve their
    # values using to_py.
    arg0_buffer.copy_to_host_async()
    arg0_buffer.copy_to_host_async()  # Duplicate calls don't do anything.
    arg1_buffer.copy_to_host_async()
    np.testing.assert_equal(arg0, arg0_buffer.to_py())
    np.testing.assert_equal(arg1, arg1_buffer.to_py())
    # copy_to_host_async does nothing after to_py is called.
    arg0_buffer.copy_to_host_async()
    np.testing.assert_equal(arg0, arg0_buffer.to_py())

  def testIpuPjRtclient__buffer__on_device_size_in_bytes(self):
    arg0 = np.array([])
    arg1 = np.array([[0., 1., 2.]], np.float32)
    arg2 = np.array([[3., 4., 5.]], np.float16)
    arg0_buffer = self.backend.buffer_from_pyval(arg0)
    arg1_buffer = self.backend.buffer_from_pyval(arg1)
    arg2_buffer = self.backend.buffer_from_pyval(arg2)
    self.assertEqual(arg0_buffer.on_device_size_in_bytes(), 0)
    # IPU size. Should align with CPU?
    self.assertEqual(arg1_buffer.on_device_size_in_bytes(), 3 * 4)
    self.assertEqual(arg2_buffer.on_device_size_in_bytes(), 3 * 2)

  def testIpuPjRtclient__buffer__numpy_array_zero_copy(self):
    # IPU buffer on HOST should be zero-copy by default.
    indata = np.array([[0., 1., 2.]], np.float32)
    buffer = self.backend.buffer_from_pyval(indata)
    outdata = np.asarray(buffer)
    self.assertEqual(outdata.ctypes.data, indata.ctypes.data)
    self.assertEqual(buffer.unsafe_buffer_pointer(), indata.ctypes.data)


class IpuPjrtClientExecutableModelTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # IPU model with 1 device.
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True,
        ipu_model_num_tiles=4,
        ipu_model_version="ipu2",
        execute_on_host_flops_limit=0.0,
    )
    cls.backend = get_ipu_client(asynchronous=async_backend, ipu_options=ipu_options)

  @classmethod
  def tearDownClass(cls):
    # Force to delete the IPU client.
    del cls.backend
    gc.collect()

  def testIpuPjRtclient__executable__successful_xla_compilation(self):
    c = xla_client.XlaBuilder(self.id())
    arg0 = np.array([10, 15, -2, 7], dtype=np.float32)
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    ops.Neg(p0)
    executable = self.backend.compile(c.build())
    self.assertIsInstance(executable, xla_extension.Executable)
    self.assertIsNone(executable.fingerprint)
    self.assertEqual(executable.client.platform, "ipu")
    self.assertEqual(executable.local_devices(), self.backend.devices())

  def testIpuPjRtclient__executable__successful_mlir_compilation_and_run(self):
    mlir_module = """
    module @jit_fn0.0 {
      func.func public @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
        %0 = mhlo.multiply %arg0, %arg0 : tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    }
    """
    c = xe.mlir.mlir_module_to_xla_computation(mlir_module, use_tuple_args=False)
    executable = self.backend.compile(c)

    self.assertIsInstance(executable, xla_extension.Executable)
    self.assertIsNone(executable.fingerprint)
    self.assertEqual(executable.client.platform, "ipu")
    self.assertEqual(executable.local_devices(), self.backend.devices())

    arg0 = np.array([-2, 7], dtype=np.float32)
    outputs = xla_client.execute_with_python_values(
        executable, [arg0], backend=self.backend
    )
    self.assertEqual(len(outputs), 1)
    np.testing.assert_equal(outputs[0], arg0 * arg0)

  @parameterized.parameters([
      [np.array(5, dtype=np.float32)],  # IPU XLA compiler optimizing to HOST.
      [np.array([10, 15, -2, 7], dtype=np.float32)],  # Running on IPU device.
  ])
  def testIpuPjRtclient__executable__unary_op__multi_executing(self, arg0):
    c = xla_client.XlaBuilder(self.id())
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    ops.Neg(p0)
    executable = self.backend.compile(c.build())

    # First run: loading on device & execute
    outputs = xla_client.execute_with_python_values(
        executable, [arg0], backend=self.backend
    )
    self.assertEqual(len(outputs), 1)
    np.testing.assert_equal(outputs[0], -arg0)
    # Second run: only executing.
    outputs = xla_client.execute_with_python_values(
        executable, [arg0], backend=self.backend
    )
    np.testing.assert_equal(outputs[0], -arg0)

  def testIpuPjRtclient__executable__binary_op__multi_executing(self):
    c = xla_client.XlaBuilder(self.id())
    arg0 = np.array([10, 15, -2, 7], dtype=np.float32)
    arg1 = np.array([1, 3, -7, 9], dtype=np.float32)
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
    ops.Mul(p0, p1)
    executable = self.backend.compile(c.build())

    # First run: loading on device & execute
    outputs = xla_client.execute_with_python_values(
        executable, [arg0, arg1], backend=self.backend
    )
    self.assertEqual(len(outputs), 1)
    np.testing.assert_equal(outputs[0], arg0 * arg1)
    # Second run: only executing.
    outputs = xla_client.execute_with_python_values(
        executable, [arg0, arg1], backend=self.backend
    )
    np.testing.assert_equal(outputs[0], arg0 * arg1)

  def testIpuPjRtclient__executable__zero_flops__executed_on_host(self):
    c = xla_client.XlaBuilder(self.id())
    arg0 = np.array([10, 15, -2, 7], dtype=np.float32)
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    ops.SliceInDim(p0, 1, 3, 1, 0)
    # ~benchmark of compilation time => good measure of compilation for IPU or CPU.
    start_compile = time.time()
    executable = self.backend.compile(c.build())
    end_compile = time.time()

    # Less than 0.1s => can only be host compilation.
    self.assertLessEqual(end_compile - start_compile, 0.1)
    # Check result!
    outputs = xla_client.execute_with_python_values(
        executable, [arg0], backend=self.backend
    )
    self.assertEqual(len(outputs), 1)
    npt.assert_equal(outputs[0], arg0[1:3])

  def testIpuPjRtclient__donated_arguments__successful_compilation(self):
    mlir_module = """
    module @jit_fn.1 {
      func.func public @main(%arg0: tensor<2xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = mhlo.add %arg0, %arg1 : tensor<2xf32>
        %1 = mhlo.subtract %arg0, %arg1 : tensor<2xf32>
        return %0, %1 : tensor<2xf32>, tensor<2xf32>
      }
    }
    """
    c = xe.mlir.mlir_module_to_xla_computation(mlir_module, use_tuple_args=False)
    self.backend.compile(c)

  def testIpuPjRtclient__donated_arguments__successful_multi_runs(self):
    mlir_module = """
    module @jit_fn.1 {
      func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = mhlo.add %arg0, %arg1 : tensor<2xf32>
        %1 = mhlo.multiply %arg0, %arg1 : tensor<2xf32>
        return %0, %1 : tensor<2xf32>, tensor<2xf32>
      }
    }
    """
    c = xe.mlir.mlir_module_to_xla_computation(mlir_module, use_tuple_args=False)
    executable = self.backend.compile(c)

    data0 = np.array([-2, 7], dtype=np.float32)
    data1 = np.array([10, 15], dtype=np.float32)
    buf0 = self.backend.buffer_from_pyval(data0)
    buf1 = self.backend.buffer_from_pyval(data1)
    # Proper first streaming of data to IPU.
    out0, out1 = executable.execute([buf0, buf1])
    # Original inputs on host => not deleted.
    self.assertFalse(buf0.is_deleted())
    self.assertFalse(buf1.is_deleted())
    # Check result streamed back to HOST.
    npt.assert_array_equal(out0, data0 + data1)
    # Check result on SRAM can be streamed back.
    out1_data = np.array(out1, copy=True)
    npt.assert_array_equal(out1_data, data0 * data1)
    # Should not be deleted in we ask sync. with HOST!
    self.assertFalse(out1.is_deleted())

    # Second run: should be re-using value already on IPU SRAM.
    out20, out21 = executable.execute([out0, out1])
    # Check input buffers status: donated buffer should have been deleted.
    # TODO: check buffers status & location.
    self.assertFalse(out0.is_deleted())
    self.assertTrue(out1.is_deleted())
    # Check result streamed back to HOST.
    self.assertFalse(out20.is_deleted())
    self.assertFalse(out21.is_deleted())
    npt.assert_array_equal(out20, data0 + data1 + (data0 * data1))
    npt.assert_array_equal(out21, (data0 + data1) * (data0 * data1))
    # Should not be deleted in we ask sync. with HOST!
    self.assertFalse(out21.is_deleted())

  def testIpuPjRtclient__donated_arguments__multi_executables(self):
    mlir_module = """
    module @jit_fn.1 {
      func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<2xf32>, tensor<2xf32>) {
        %0 = mhlo.add %arg0, %arg1 : tensor<2xf32>
        %1 = mhlo.multiply %arg0, %arg1 : tensor<2xf32>
        return %0, %1 : tensor<2xf32>, tensor<2xf32>
      }
    }
    """
    c = xe.mlir.mlir_module_to_xla_computation(mlir_module, use_tuple_args=False)
    executable0 = self.backend.compile(c)
    executable1 = self.backend.compile(c)

    data0 = np.array([-2, 7], dtype=np.float32)
    data1 = np.array([10, 15], dtype=np.float32)
    buf0 = self.backend.buffer_from_pyval(data0)
    buf1 = self.backend.buffer_from_pyval(data1)
    # First executable called.
    out00, out01 = executable0.execute([buf0, buf1])
    npt.assert_array_equal(out00, data0 + data1)

    # Second executable called.
    out10, out11 = executable1.execute([buf0, out00])
    npt.assert_array_equal(out10, data0 + (data0 + data1))
    # First on-device buffer should be deleted (even if not directly used/overwritten).
    self.assertTrue(out01.is_deleted())
    self.assertFalse(out11.is_deleted())
    npt.assert_array_equal(out11, data0 * (data0 + data1))

  def testIpuPjRtclient__donated_arguments__multi_inputs_outputs(self):
    mlir_module = """
    module @jit_fn.1 {
      func.func public @main(%arg0: tensor<2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
        %0 = mhlo.add %arg0, %arg1 : tensor<2xf32>
        %1 = mhlo.subtract %arg0, %arg1 : tensor<2xf32>
        %2 = mhlo.multiply %arg0, %arg1 : tensor<2xf32>
        return %0, %1, %2 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
      }
    }
    """
    c = xe.mlir.mlir_module_to_xla_computation(mlir_module, use_tuple_args=False)
    executable = self.backend.compile(c)

    data0 = np.array([-2, 7], dtype=np.float32)
    data1 = np.array([10, 15], dtype=np.float32)
    buf0 = self.backend.buffer_from_pyval(data0)
    buf1 = self.backend.buffer_from_pyval(data1)
    # Proper first streaming of data to IPU.
    out00, out01, out02 = executable.execute([buf0, buf1])
    npt.assert_array_equal(out00, data0 + data1)
    npt.assert_array_equal(out01, data0 - data1)
    npt.assert_array_equal(out02, data0 * data1)

    out10, out11, out12 = executable.execute([buf0, out02])
    self.assertTrue(out01.is_deleted())
    self.assertTrue(out02.is_deleted())
    self.assertFalse(out11.is_deleted())
    self.assertFalse(out12.is_deleted())
    npt.assert_array_equal(out10, data0 + (data0 * data1))

  @unittest.skipUnless(async_backend, "Requires asynchronous IPU backend.")
  def testIpuPjRtclient__executable__asynchronous_dispatch__timing_no_data_dependency(
      self
  ):
    # Large enough input such that computation is not trivially fast.
    N = 100000
    data = np.arange(N).astype(np.float32)
    # Simple compute graph.
    c = xla_client.XlaBuilder(self.id())
    p = ops.Parameter(c, 0, xla_client.shape_from_pyval(data))
    ops.Mul(p, p)
    executable = self.backend.compile(c.build())

    num_iters = 30
    input0 = self.backend.buffer_from_pyval(data)
    async_timings = approximate_timer(lambda: executable.execute([input0]), num_iters)
    block_timings = approximate_timer(
        lambda: executable.execute([input0])[0].block_until_ready(), num_iters
    )
    # Async. dispatch <= 50us
    self.assertLessEqual(np.median(async_timings), 5 * 1e-5)
    # Async. dispatch should be at least 10x faster.
    self.assertGreaterEqual(np.median(block_timings), 10 * np.median(async_timings))

  @unittest.skipUnless(async_backend, "Requires asynchronous IPU backend.")
  def testIpuPjRtclient__executable__asynchronous_dispatch__inout_dependency(self):
    # Large enough input such that computation is not trivially fast.
    N = 1000000
    data = np.arange(N).astype(np.float32)
    # Simple compute graph.
    c = xla_client.XlaBuilder(self.id())
    p = ops.Parameter(c, 0, xla_client.shape_from_pyval(data))
    ops.Mul(p, p)

    executable = self.backend.compile(c.build())
    inputs = [self.backend.buffer_from_pyval(data)]
    num_iters = 30
    # "pre-warming"
    inputs = executable.execute(inputs)
    inputs = executable.execute(inputs)
    inputs[0].block_until_ready()

    start = time.perf_counter()
    for _ in range(num_iters):
      # NOTE: erasing previous buffer! Checking usage hold.
      # Classic "training" loop pattern!
      inputs = executable.execute(inputs)
    async_timing = time.perf_counter() - start
    inputs[0].block_until_ready()

    start = time.perf_counter()
    for _ in range(num_iters):
      inputs = executable.execute(inputs)
      inputs[0].block_until_ready()
    block_timing = time.perf_counter() - start

    # Checking async. dispatch much faster than blocking (and <50us).
    self.assertLessEqual(async_timing / num_iters, 5 * 1e-5)
    self.assertLessEqual(async_timing * 10, block_timing)


if __name__ == "__main__":
  absltest.main()
