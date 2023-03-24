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
"""IPU PRJT device API unit tests."""
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

# Skipping some tests if no local IPU hardware.
ipu_hw_available = IpuDeviceMeshManager.has_local_ipu_hardware()


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

  def testIpuPjRtclient__make_ipu_pjrt_options__from_flags(self):
    flags = {
        'always_rearrange_copies_on_the_host': 'true',
        'device_count': '2',
        'model_num_tiles': '8',
        'num_io_tiles': '10',
        'use_legacy_client': 'true',
        'use_model': 'true',
        'execute_on_host_flops_limit': '2.0',
    }
    ipu_options = make_ipu_pjrt_options(flags)
    self.assertIsInstance(ipu_options, IpuPjRtOptions)
    self.assertTrue(ipu_options.use_ipu_model)
    self.assertEqual(ipu_options.ipu_model_num_tiles, 8)
    self.assertTrue(ipu_options.always_rearrange_copies_on_the_host)
    self.assertIsNone(ipu_options.visible_devices)
    self.assertEqual(ipu_options.execute_on_host_flops_limit, 2)

  def testIpuPjRtclient__make_ipu_client__from_env_variables(self):
    env = {
        "XLA_IPU_PLATFORM_DEVICE_COUNT": "1",
        "XLA_IPU_PLATFORM_ALWAYS_REARRANGE_COPIES_ON_THE_HOST": "false",
        "JAX_IPU_USE_MODEL": "true",
        "JAX_IPU_MODEL_NUM_TILES": "8",
        "JAX_IPU_USE_LEGACY_CLIENT": "false",
        "XLA_IPU_PLATFORM_NUM_IO_TILES": "0",
        "PATH": "blalba"
    }
    client = make_ipu_client(env)
    self.assertIsInstance(client, xla_client.Client)


class IpuPjrtClientBaseTest(parameterized.TestCase):

  def setUp(self):
    super(IpuPjrtClientBaseTest, self).setUp()

  def testIpuPjRtclient__get_ipu_client__base_properties(self):
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=16, ipu_model_version="ipu21"
    )
    ipu_client = get_ipu_client(True, ipu_options)
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
    ipu_client = get_ipu_client(True, ipu_options)
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

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__ipu_hardware_local_devices(self):
    ipu_client = get_ipu_client(True, IpuPjRtOptions())
    ipu_devices = ipu_client.local_devices()

    self.assertGreater(len(ipu_devices), 0)
    self.assertEqual([d.id for d in ipu_devices], list(range(len(ipu_devices))))
    self.assertEqual({d.num_tiles for d in ipu_devices}, {1472})
    self.assertEqual({d.version for d in ipu_devices}, {"ipu2"})
    self.assertEqual({d.type for d in ipu_devices}, {IpuTargetType.IPU})

  @unittest.skipIf(not ipu_hw_available, "No IPU hardware available.")
  def testIpuPjRtclient__get_ipu_client__multi_clients_create_delete(self):
    ipu_client = get_ipu_client(True, IpuPjRtOptions())
    # Check resources are properly freed.
    del ipu_client
    gc.collect()
    get_ipu_client(True, IpuPjRtOptions())


class IpuPjrtClientBufferTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    # IPU model with 1 device.
    ipu_options = IpuPjRtOptions(
        use_ipu_model=True, ipu_model_num_tiles=4, ipu_model_version="ipu2"
    )
    cls.backend = get_ipu_client(True, ipu_options)

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
    cls.backend = get_ipu_client(True, ipu_options)

  @classmethod
  def tearDownClass(cls):
    # Force to delete the IPU client.
    del cls.backend
    gc.collect()

  def testIpuPjRtclient__executable__successful_compilation(self):
    c = xla_client.XlaBuilder(self.id())
    arg0 = np.array([10, 15, -2, 7], dtype=np.float32)
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    ops.Neg(p0)
    executable = self.backend.compile(c.build())
    self.assertIsInstance(executable, xla_extension.Executable)
    self.assertIsNone(executable.fingerprint)
    self.assertEqual(executable.client.platform, "ipu")
    self.assertEqual(executable.local_devices(), self.backend.devices())

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


if __name__ == "__main__":
  absltest.main()
