# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
"""Backend-dependent tests for the Python XLA client."""

import functools
import itertools
import re
import threading
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import make_ipu_client, IpuDevice, IpuTargetType, get_ipu_client, IpuPjRtDevice, IpuPjRtOptions

bfloat16 = xla_client.bfloat16
ops = xla_client.ops
FLAGS = flags.FLAGS

# We choose to ignore pylint's complaints about complex comprehensions, which we
# use widely for parameterizing tests.
# pylint: disable=g-complex-comprehension


def make_parameter(c, arr, idx=0):
  """Make ops parameter, necessary for IPU testing.
  """
  # We need to modify slightly the official XLA client unit tests:
  # 1. Poplar Compiler will Skip engine compilation if output is constant(compute by CPU)， So test as Parameter input rather than constant
  # 2. Parameter's shape with layout will cause an check error when IPU, so call with_major_to_minor_layout_if_absent to ge a default layout
  return ops.Parameter(
      c, idx,
      xla_client.shape_from_pyval(arr).with_major_to_minor_layout_if_absent()
  )


def make_parameters(c, arrs):
  return [make_parameter(c, arr, idx) for idx, arr in enumerate(arrs)]


def TestFactory(xla_backend):
  """Tests are ported from `xla_client_test.py`, and adapted for IPU running (mainly
  replacing `constant` by `parameter` to force running on IPU backend).
  """

  tests = []
  cloud_tpu = False
  ipu_backend = True
  # IPU supported dtypes.
  int_dtypes = [np.int32]
  float_dtypes = [np.float32]

  class ComputationTest(parameterized.TestCase):
    """Base class for running an XLA Computation through the local client."""

    def setUp(self):
      super(ComputationTest, self).setUp()
      self.backend = xla_backend()

    def _NewComputation(self, name=None):
      if name is None:
        name = self.id()
      return xla_client.XlaBuilder(name)

    def _Execute(self, c, arguments):
      compiled_c = self.backend.compile(c.build())
      return xla_client.execute_with_python_values(
          compiled_c, arguments, backend=self.backend
      )

    def _ExecuteAndAssertWith(self, assert_func, c, arguments, expected):
      assert expected is not None
      results = self._Execute(c, arguments)
      self.assertLen(results, len(expected))
      for result, e in zip(results, expected):
        # Numpy's comparison methods are a bit too lenient by treating inputs as
        # "array-like", meaning that scalar 4 will be happily compared equal to
        # [[4]]. We'd like to be more strict so assert shapes as well.
        self.assertEqual(np.asanyarray(result).shape, np.asanyarray(e).shape)
        assert_func(result, e)

    def _ExecuteAndCompareExact(self, c, arguments=(), expected=None):
      self._ExecuteAndAssertWith(np.testing.assert_equal, c, arguments, expected)

    def _ExecuteAndCompareClose(
        self, c, arguments=(), expected=None, rtol=1e-4, atol=0
    ):
      self._ExecuteAndAssertWith(
          functools.partial(np.testing.assert_allclose, rtol=rtol, atol=atol), c,
          arguments, expected
      )

  def NumpyArrayF32(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.float32 dtype."""
    return np.array(*args, dtype=np.float32, **kwargs)

  def NumpyArrayF64(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.float64 dtype."""
    return np.array(*args, dtype=np.float64, **kwargs)

  def NumpyArrayS32(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.int32 dtype."""
    return np.array(*args, dtype=np.int32, **kwargs)

  def NumpyArrayBool(*args, **kwargs):
    """Convenience wrapper to create Numpy arrays with a np.bool_ dtype."""
    return np.array(*args, dtype=np.bool_, **kwargs)

  class IpuBackendClientTest(ComputationTest):
    """IPU backend/client tests."""

    def testIpuDevicePlatform(self):
      device = self.backend.local_devices()[0]
      assert self.backend.platform == "ipu"
      assert device.platform == "ipu"
      assert device.version in {"ipu2", "ipu21"}
      assert isinstance(device, IpuPjRtDevice)
      assert repr(
          device
      ) == f"IpuDevice(id={device.id}, num_tiles={device.num_tiles}, version={device.version})"

    def testIpuDeviceHardwareProperties(self):
      device = self.backend.local_devices()[0]
      assert device.target_type in {IpuTargetType.IPU_MODEL, IpuTargetType.IPU}
      if device.target_type == IpuTargetType.IPU:
        assert device.num_tiles == 1472
      if device.target_type == IpuTargetType.IPU_MODEL:
        assert device.num_tiles in {4, 8}
      assert device.num_worker_contexts == 6
      assert device.bytes_per_tile == 638976
      assert device.tile_clock_frequency == 1850000000

  tests.append(IpuBackendClientTest)

  class ComputationPrinting(absltest.TestCase):

    def setUp(self):
      super(ComputationPrinting, self).setUp()
      self.backend = xla_backend()

    def ExampleComputation(self):
      builder = xla_client.XlaBuilder("acomputation")
      p0 = ops.Parameter(builder, 0, xla_client.shape_from_pyval(np.float32(0)))
      p1 = ops.Parameter(
          builder, 1, xla_client.shape_from_pyval(np.zeros((4,), np.float32))
      )
      x = ops.Mul(p0, p1)
      ops.Add(x, x)
      return builder.build()

    @unittest.skipIf(ipu_backend, "Failing on IPU.")
    @unittest.skipIf(cloud_tpu, "not implemented")
    def testCompiledHloModuleToHloText(self):
      computation = self.ExampleComputation()
      executable = self.backend.compile(computation)
      hlo_modules = executable.hlo_modules()
      self.assertLen(hlo_modules, 1)
      hlo_text = hlo_modules[0].to_string()
      self.assertTrue(hlo_text.startswith("HloModule acomputation"))
      self.assertIn("fusion", hlo_text)

    @unittest.skipIf(ipu_backend, "Failing on IPU.")
    @unittest.skipIf(cloud_tpu, "not implemented")
    def testCompiledHloModuleAsSerializedProto(self):
      computation = self.ExampleComputation()
      executable = self.backend.compile(computation)
      hlo_modules = executable.hlo_modules()
      self.assertLen(hlo_modules, 1)
      hlo_text = hlo_modules[0].to_string()
      proto = hlo_modules[0].as_serialized_hlo_module_proto()
      hlo_module_roundtrip = xla_client.XlaComputation(proto).get_hlo_module()
      hlo_text_roundtrip = hlo_module_roundtrip.to_string()
      self.assertEqual(hlo_text, hlo_text_roundtrip)

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testStableComputationSerialization(self):
      # Ideally we would test identical computations produced in different
      # processes. For now we have this limited smoke test.
      computation = self.ExampleComputation()
      ref = computation.as_serialized_hlo_module_proto()
      for _ in range(10):
        self.assertEqual(computation.as_serialized_hlo_module_proto(), ref)

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testFlopEstimate(self):
      computation = self.ExampleComputation()
      properties = xla_client._xla.hlo_module_cost_analysis(
          self.backend, computation.as_hlo_module()
      )
      self.assertEqual(properties["flops"], 8.0)

    def testFingerprint(self):
      computation = self.ExampleComputation()
      executable = self.backend.compile(computation)
      fingerprint = executable.fingerprint
      if self.backend.platform == "tpu" and not cloud_tpu:
        logging.info("fingerprint: %s", fingerprint)
        self.assertNotEmpty(fingerprint)
      else:
        self.assertIsNone(fingerprint)

  tests.append(ComputationPrinting)

  class ComputationsWithConstantsTest(ComputationTest):
    """IPU backend skipping constant ops."""

  tests.append(ComputationsWithConstantsTest)

  class PythonCallbackTest(ComputationTest):

    @unittest.skipIf(ipu_backend, "Not implemented on IPU backend.")
    def testPythonCallback(self):
      if self.backend.platform not in {"cpu", "gpu", "ipu"}:
        self.skipTest("Test requires cpu or gpu platform")
      c = self._NewComputation()

      f = lambda x, y: (x + y, x - y)

      arg0 = np.array([9, 43, -101, 22], dtype=np.int32)
      arg1 = np.array([10, 15, -2, 7], dtype=np.int32)
      shape = xla_client.shape_from_pyval(arg0)
      shape = shape.with_major_to_minor_layout_if_absent()
      p0 = ops.Parameter(c, 0, shape)
      p1 = ops.Parameter(c, 1, shape)
      out, keepalive = self.backend.emit_python_callback(f, c, [p0, p1], [shape, shape])
      self._ExecuteAndCompareExact(
          c, arguments=[arg0, arg1], expected=[arg0 + arg1, arg0 - arg1]
      )
      del out, keepalive

    @unittest.skipIf(ipu_backend, "Not implemented on IPU backend.")
    def testPythonCallbackCanHandleExceptions(self):
      if self.backend.platform not in {"cpu", "gpu", "ipu"}:
        self.skipTest("Test requires cpu or gpu platform")
      c = self._NewComputation()

      def _Callback(x):
        raise ValueError("Value error raised!")

      arg0 = np.array([9, 43, -101, 22], dtype=np.int32)
      shape = xla_client.shape_from_pyval(arg0)
      shape = shape.with_major_to_minor_layout_if_absent()
      p0 = ops.Parameter(c, 0, shape)
      out, keepalive = self.backend.emit_python_callback(
          _Callback, c, [p0], [shape], has_side_effects=True
      )
      with self.assertRaisesRegex(xla_client.XlaRuntimeError, "Value error raised!"):
        self._Execute(c, [arg0])
      del out, keepalive

    @unittest.skipIf(ipu_backend, "Not implemented on IPU backend.")
    def testTokens(self):
      if self.backend.platform not in {"cpu", "gpu", "ipu"}:
        self.skipTest("Test requires cpu or gpu platform")
      c = self._NewComputation()

      def _Callback(x, y):
        assert y is None, y
        return None, x + 1

      arg0 = np.array([9, 43, -101, 22], dtype=np.int32)
      shape = xla_client.shape_from_pyval(arg0)
      token_shape = xla_client.Shape.token_shape()
      p0 = ops.Parameter(c, 0, shape)
      token = ops.CreateToken(c)
      out, keepalive = self.backend.emit_python_callback(
          _Callback, c, [p0, token], [token_shape, shape]
      )
      out = ops.GetTupleElement(out, 1)
      self._ExecuteAndCompareExact(c, arguments=[arg0], expected=[arg0 + 1])
      del out, keepalive

    @unittest.skipIf(ipu_backend, "Not implemented on IPU backend.")
    def testStriding(self):
      if self.backend.platform not in {"cpu", "gpu", "ipu"}:
        self.skipTest("Test requires cpu or gpu platform")
      c = self._NewComputation()

      def _Callback(x):
        assert x.flags.f_contiguous, x.strides
        # Force the output array to have C layout, which will require a
        # transpose back to the expected Fortran layout.
        return np.ascontiguousarray(x * 2),

      arg0 = np.arange(12, dtype=np.int16).reshape(3, 4)
      shape_f_layout = xla_client.Shape.array_shape(
          arg0.dtype, arg0.shape, layout=(0, 1)
      )
      p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
      out, keepalive = self.backend.emit_python_callback(
          _Callback, c, [p0], [shape_f_layout], [shape_f_layout]
      )
      self._ExecuteAndCompareExact(c, arguments=[arg0], expected=[arg0 * 2])
      del out, keepalive

  tests.append(PythonCallbackTest)

  class ParametersTest(ComputationTest):
    """Tests focusing on Parameter ops and argument-passing."""

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in int_dtypes)
    def testScalarTimesVector(self, dtype):
      c = self._NewComputation()
      arg0 = np.array(3, dtype=dtype)
      arg1 = np.array([10, 15, -2, 7], dtype=dtype)
      p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
      p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
      ops.Mul(p0, p1)
      self._ExecuteAndCompareExact(c, arguments=[arg0, arg1], expected=[arg0 * arg1])

    # TODO(phawkins): test comparison harness doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testScalarMinusVectorExplicitNumbering(self, dtype):
      # Use explicit numbering and pass parameter_num first. Sub is used since
      # it's not commutative and can help catch parameter reversal within the
      # computation.
      c = self._NewComputation()
      arg0 = np.array(2.0, dtype=dtype)
      arg1 = np.array([-2.3, 3.3, -4.3, 5.3], dtype=dtype)
      p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
      p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
      ops.Sub(p1, p0)
      self._ExecuteAndCompareClose(c, arguments=[arg0, arg1], expected=[arg1 - arg0])

  tests.append(ParametersTest)

  class BufferTest(ComputationTest):
    """Tests focusing on execution with Buffers."""

    def testConstantSum(self):
      c = self._NewComputation()
      ops.Add(ops.Constant(c, np.float32(1.11)), ops.Constant(c, np.float32(3.14)))
      self._ExecuteAndCompareClose(c, expected=[4.25])

    def testOneParameterSum(self):
      # NOTE: getting executed on HOST as scalar graph.
      c = self._NewComputation()
      ops.Add(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
          ops.Constant(c, np.float32(3.14))
      )
      self._ExecuteAndCompareClose(c, arguments=[NumpyArrayF32(1.11)], expected=[4.25])

    def testTwoParameterSum(self):
      # NOTE: getting executed on HOST as scalar graph.
      c = self._NewComputation()
      ops.Add(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.0))),
          ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0.0)))
      )
      self._ExecuteAndCompareClose(
          c, arguments=[NumpyArrayF32(1.11), NumpyArrayF32(3.14)], expected=[4.25]
      )

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testCannotCallWithDeletedBuffers(self):
      c = self._NewComputation()
      ops.Add(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0.))),
          ops.Constant(c, np.float32(3.14))
      )
      arg = NumpyArrayF32(1.11)
      compiled_c = self.backend.compile(c.build())
      arg_buffer = self.backend.buffer_from_pyval(arg)
      arg_buffer.delete()
      with self.assertRaises(xla_client.XlaRuntimeError):
        compiled_c.execute([arg_buffer])

    def testXlaShape(self):
      pyval = np.array([[1., 2.]], np.float32)
      local_buffer = self.backend.buffer_from_pyval(pyval)
      xla_shape = local_buffer.xla_shape()
      self.assertEqual(xla_shape.dimensions(), (1, 2))
      self.assertEqual(np.dtype(xla_shape.element_type()), np.dtype(np.float32))

    def testXlaShapeIndex(self):
      a = xla_client.ShapeIndex((1, 2))
      b = xla_client.ShapeIndex((1, 2))
      c = xla_client.ShapeIndex((2, 3))
      self.assertEqual(a, b)
      self.assertNotEqual(b, c)

    def testLayout(self):
      f32 = xla_client.PrimitiveType.F32
      a = xla_client.Shape.array_shape(f32, (2, 3), (0, 1)).layout()
      b = xla_client.Shape.array_shape(f32, (2, 3), (0, 1)).layout()
      c = xla_client.Shape.array_shape(f32, (2, 3), (1, 0)).layout()
      self.assertEqual(a.minor_to_major(), (0, 1))
      self.assertEqual(b.minor_to_major(), (0, 1))
      self.assertEqual(c.minor_to_major(), (1, 0))
      self.assertEqual(a, b)
      self.assertNotEqual(a, c)
      self.assertNotEqual(b, c)
      self.assertEqual(hash(a), hash(b))
      self.assertNotEqual(hash(a), hash(c))
      self.assertNotEqual(hash(b), hash(c))

    def testBlockUntilReadyWorks(self):
      arg = np.array([[1., 2.]], np.float32)
      arg_buffer = self.backend.buffer_from_pyval(arg)
      arg_buffer.block_until_ready()
      # This test merely checks that nothing goes awry when we call
      # block_until_ready(); it's difficult to test anything else.

    def testBlockUntilReadyRaisesOnDeletedBuffer(self):
      arg = np.array([[1., 2.]], np.float32)
      buffer = self.backend.buffer_from_pyval(arg)
      buffer.delete()
      with self.assertRaisesRegex(
          RuntimeError,
          re.escape("BlockHostUntilReady() called on deleted or donated buffer")
      ):
        buffer.block_until_ready()

    def testDeviceArrayBaseSignatures(self):
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

    def testOnDeviceSizeInBytes(self):
      if not isinstance(self.backend, xla_client.Client):
        self.skipTest("TPU Driver doesn't support OnDeviceSizeInBytes.")
      arg0 = np.array([])
      arg1 = np.array([[0., 1., 2.]], np.float32)
      arg2 = np.array([[3., 4., 5.]], bfloat16)
      arg0_buffer = self.backend.buffer_from_pyval(arg0)
      arg1_buffer = self.backend.buffer_from_pyval(arg1)
      arg2_buffer = self.backend.buffer_from_pyval(arg2)
      self.assertEqual(arg0_buffer.on_device_size_in_bytes(), 0)
      # OnDeviceSizeInBytes varies depending on the platform. Confirm there's
      # a reasonable value.
      self.assertGreater(arg1_buffer.on_device_size_in_bytes(), 0)
      self.assertGreater(arg2_buffer.on_device_size_in_bytes(), 0)

    def testLiveBuffers(self):
      if not isinstance(self.backend, xla_client.Client):
        self.skipTest("TPU Driver doesn't support LiveBuffers().")
      self.assertEmpty(self.backend.live_buffers())
      arg0 = np.array([])
      arg1 = np.array([[0., 1., 2.]], np.float32)
      arg2 = np.array([[3., 4., 5.]], bfloat16)
      arg0_buffer = self.backend.buffer_from_pyval(arg0)
      arg1_buffer = self.backend.buffer_from_pyval(arg1)
      arg2_buffer = self.backend.buffer_from_pyval(arg2)
      self.assertLen(self.backend.live_buffers(), 3)
      self.assertIs(self.backend.live_buffers()[0], arg2_buffer)
      self.assertIs(self.backend.live_buffers()[1], arg1_buffer)
      self.assertIs(self.backend.live_buffers()[2], arg0_buffer)
      self.assertEqual(
          self.backend.devices()[0].live_buffers(), self.backend.live_buffers()
      )

      arg1_buffer.delete()
      self.assertLen(self.backend.live_buffers(), 2)
      self.assertIs(self.backend.live_buffers()[0], arg2_buffer)
      self.assertIs(self.backend.live_buffers()[1], arg0_buffer)

      arg0_buffer.delete()
      arg2_buffer.delete()
      self.assertEmpty(self.backend.live_buffers())

    def testCopyToHost(self):
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

    def testDevice(self):
      x = np.arange(8, dtype=np.int32)
      for device in self.backend.local_devices():
        buf = self.backend.buffer_from_pyval(x, device=device)
        self.assertEqual(buf.device(), device)
        np.testing.assert_equal(x, buf.to_py())

    def testStandardTypes(self):
      standard_dtypes = [np.int8, np.int16, np.int32, np.float16, np.float32]
      for dtype in standard_dtypes:
        if dtype == bfloat16 or dtype == np.complex128:
          continue
        arr = self.backend.buffer_from_pyval(np.array([0, 1], dtype))
        arr = arr.to_py()
        self.assertEqual(dtype, type(arr[0]))

    def testUnsafeBufferPointer(self):
      if not isinstance(self.backend, xla_client.Client):
        self.skipTest("TPU Driver doesn't support UnsafeBufferPointer().")
      arg0 = np.array([])
      arg1 = np.array([[0., 1., 2.]], np.float32)
      arg2 = np.array([[3., 4., 5.]], bfloat16)
      arg0_buffer = self.backend.buffer_from_pyval(arg0)
      arg1_buffer = self.backend.buffer_from_pyval(arg1)
      arg2_buffer = self.backend.buffer_from_pyval(arg2)
      self.assertGreaterEqual(arg0_buffer.unsafe_buffer_pointer(), 0)
      self.assertGreaterEqual(arg1_buffer.unsafe_buffer_pointer(), 0)
      self.assertGreaterEqual(arg2_buffer.unsafe_buffer_pointer(), 0)

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testClone(self):
      x = np.array([[3., 4., 5.]], np.float32)
      y = self.backend.buffer_from_pyval(x)
      z = y.clone()
      self.assertNotEqual(id(x), id(y))
      np.testing.assert_array_equal(y.to_py(), z.to_py())
      self.assertEqual(y.unsafe_buffer_pointer(), z.unsafe_buffer_pointer())

    @unittest.skipIf(cloud_tpu, "not implemented")
    def testJaxAttributesHaveCorrectDefaults(self):
      x = np.array([[3., 4., 5.]], np.float32)
      y = self.backend.buffer_from_pyval(x)
      self.assertIsNone(y.aval)
      self.assertIsNone(y._device)

  tests.append(BufferTest)

  # We need to modify slightly the official XLA client unit tests:
  # 1. Poplar Compiler will Skip engine compilation if output is constant(compute by CPU)， So test as Parameter input rather than constant
  # 2. Parameter's shape with layout will cause an check error when IPU, so call with_major_to_minor_layout_if_absent to ge a default layout

  class SingleOpTest(ComputationTest):
    """Tests for single ops.

    The goal here is smoke testing - to exercise the most basic functionality of
    single XLA ops. As minimal as possible number of additional ops are added
    around the op being tested.
    """

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testConcatenate(self, dtype):
      c = self._NewComputation()
      args = (
          np.array([1.0, 2.0, 3.0], dtype=dtype),
          np.array([4.0, 5.0, 6.0], dtype=dtype),
      )
      ops.ConcatInDim(c, make_parameters(c, args), dimension=0)
      self._ExecuteAndCompareExact(
          c,
          arguments=args,
          expected=[np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)]
      )

    # pyformat: disable
    @parameterized.named_parameters(
        {
            "testcase_name": "_{}_{}".format(src_dtype.__name__, dst_dtype.__name__),
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
        } for src_dtype, dst_dtype in
        itertools.permutations([np.bool_, np.int32, np.float32], 2)
    )
    # pyformat: enable
    def testConvertElementType(self, src_dtype, dst_dtype):
      if ((src_dtype in [np.int64, np.float64] or dst_dtype in [np.int64, np.float64])
          and self.backend.platform == "tpu"):
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      x = np.array([0, 1, 0, 0, 1], dtype=src_dtype)
      ops.ConvertElementType(make_parameter(c, x), xla_client.dtype_to_etype(dst_dtype))

      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (x,), backend=self.backend
      )
      self.assertLen(result, 1)
      expected = np.array(x, dtype=dst_dtype)

      self.assertEqual(result[0].shape, expected.shape)
      self.assertEqual(result[0].dtype, expected.dtype)
      np.testing.assert_equal(result[0], expected)

    # TODO(phawkins): np.dot implementation doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testDotMatrixVector(self, dtype):
      c = self._NewComputation()
      lhs = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=dtype)
      rhs = np.array([[10.0], [20.0]], dtype=dtype)
      ops.Dot(*make_parameters(c, [lhs, rhs]))
      self._ExecuteAndCompareClose(c, arguments=[lhs, rhs], expected=[np.dot(lhs, rhs)])

  tests.append(SingleOpTest)

  class EmbeddedComputationsTest(ComputationTest):
    """Tests for XLA graphs with embedded computations (such as maps)."""

    def _CreateConstantComputation(self, in_dtype, out_dtype):
      """Computation (A) -> B that returns a constant 1 for any input."""
      c = self._NewComputation(
          "constant_{}_{}_one".format(in_dtype.__name__, out_dtype.__name__)
      )
      ops.Parameter(
          c, 0,
          xla_client.shape_from_pyval(np.array(0, dtype=in_dtype)
                                     ).with_major_to_minor_layout_if_absent()
      )
      ops.Constant(c, out_dtype(1))
      return c.build()

    def _CreateMulBy2Computation(self, dtype):
      """Computation (dtype) -> dtype that multiplies its parameter by 2."""
      c = self._NewComputation("mul_f32_by2")
      ops.Mul(
          ops.Parameter(
              c, 0,
              xla_client.shape_from_pyval(np.array(0, dtype=dtype)
                                         ).with_major_to_minor_layout_if_absent()
          ), ops.Constant(c, dtype(2.0))
      )
      return c.build()

    def _CreateMulF32ByParamComputation(self):
      """Computation (f32) -> f32 that multiplies one parameter by the other."""
      c = self._NewComputation("mul_f32_by_param")
      ops.Mul(
          ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(0))),
          ops.Parameter(c, 1, xla_client.shape_from_pyval(NumpyArrayF32(0)))
      )
      return c.build()

    def _CreateBinaryAddComputation(self, dtype):
      """Computation (dtype, dtype) -> dtype that adds its two parameters."""
      c = self._NewComputation("add_param0_by_param1")
      shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
      shape = shape.with_major_to_minor_layout_if_absent()
      ops.Add(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
      return c.build()

    def _CreateBinaryGeComputation(self, dtype):
      """Computation (dtype, dtype) -> bool that tests param0 >= param1."""
      c = self._NewComputation("param0_lt_param1")
      shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
      shape = shape.with_major_to_minor_layout_if_absent()
      ops.Ge(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
      return c.build()

    def _MakeSample3DArray(self, dtype):
      return np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
                       [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]],
                      dtype=dtype)

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testCall(self, dtype):
      c = self._NewComputation()
      data = dtype(5.0)
      ops.Call(
          c, self._CreateMulBy2Computation(dtype), operands=(make_parameter(c, data),)
      )
      self._ExecuteAndCompareClose(c, arguments=(data,), expected=[10.0])

    @parameterized.named_parameters({
        "testcase_name": "_{}_{}".format(in_dtype.__name__, out_dtype.__name__),
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
    } for in_dtype, out_dtype in [[np.float32, np.int32]])
    def testMapEachElementToConstant(self, in_dtype, out_dtype):
      c = self._NewComputation()
      data = np.array([1.0, 2.0, 3.0, 4.0], dtype=in_dtype)
      ops.Map(
          c, [make_parameter(c, data)],
          self._CreateConstantComputation(in_dtype, out_dtype), [0]
      )
      self._ExecuteAndCompareExact(c, arguments=(data,), expected=[[1, 1, 1, 1]])

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testMapMulBy2(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      data = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      ops.Map(c, [make_parameter(c, data)], self._CreateMulBy2Computation(dtype), [0])
      self._ExecuteAndCompareClose(
          c, arguments=(data,), expected=[[2.0, 4.0, 6.0, 8.0]]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSimpleMapChain(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      # Chains a map of constant-out with a map of mul-by-2
      c = self._NewComputation()
      data = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      const = ops.Map(
          c, [make_parameter(c, data)], self._CreateConstantComputation(dtype, dtype),
          [0]
      )
      ops.Map(c, [const], self._CreateMulBy2Computation(dtype), [0])
      self._ExecuteAndCompareClose(
          c, arguments=(data,), expected=[[2.0, 2.0, 2.0, 2.0]]
      )

    # TODO(b/154752816): bfloat16 crashes in evaluator.
    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes if dtype != bfloat16)
    def testDivVectorsWithMap(self, dtype):

      def DivComputation():
        c = self._NewComputation("div_param0_by_param1")
        shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
        ops.Div(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
        return c.build()

      c = self._NewComputation()
      data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      data2 = np.array([5.0, 5.0, 4.0, 4.0], dtype=dtype)
      ops.Map(c, make_parameters(c, [data1, data2]), DivComputation(), [0])
      self._ExecuteAndCompareClose(
          c, arguments=(data1, data2), expected=[[0.2, 0.4, 0.75, 1.0]], rtol=1e-3
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testSelectAndScatter(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      data = np.array([[1., 2., 6.], [4., 5., 3.]], dtype=dtype)
      p = make_parameter(c, data)
      window_dimensions = (2, 1)
      window_strides = (1, 2)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID,
          c.get_shape(p).dimensions(), window_dimensions, window_strides
      )
      ops.SelectAndScatterWithGeneralPadding(
          p,
          select=self._CreateBinaryGeComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=padding,
          source=ops.Constant(c, np.array([[0.1, 0.2]], dtype=dtype)),
          init_value=ops.Constant(c, np.array(1, dtype=dtype)),
          scatter=self._CreateBinaryAddComputation(dtype)
      )
      self._ExecuteAndCompareClose(
          c, arguments=(data,), expected=[[[1., 1., 1.2], [1.1, 1., 1.]]], rtol=5e-3
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduce1DtoScalar(self, dtype):
      c = self._NewComputation()
      data = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
      ops.Reduce(
          c,
          operands=[make_parameter(c, data)],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=[0]
      )
      self._ExecuteAndCompareClose(c, arguments=(data,), expected=[10])

    # TODO(phawkins): test comparison harness doesn't support bfloat16
    @parameterized.named_parameters({
        "testcase_name": "_{}_dim{}".format(dtype.__name__, dim),
        "dtype": dtype,
        "dim": dim,
    } for dtype in float_dtypes if dtype != bfloat16 for dim in range(2))
    def testReduce2DTo1D(self, dtype, dim):
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[make_parameter(c, input_array)],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=[dim]
      )
      self._ExecuteAndCompareClose(
          c, arguments=(input_array,), expected=[np.sum(input_array, axis=dim)]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}_dims[{}]".format(dtype.__name__, dims),
        "dtype": dtype,
        "dims": tuple(dims)
    } for dtype in float_dtypes for dims in itertools.permutations(range(3)))
    def testReduce3DAllPossibleWaysF32(self, dtype, dims):
      input_array = self._MakeSample3DArray(dtype)
      c = self._NewComputation()
      ops.Reduce(
          c,
          operands=[make_parameter(c, input_array)],
          init_values=[ops.Constant(c, dtype(0))],
          computation=self._CreateBinaryAddComputation(dtype),
          dimensions_to_reduce=dims
      )
      self._ExecuteAndCompareClose(
          c, arguments=(input_array,), expected=[np.sum(input_array, axis=dims)]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowValidUnitStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 1)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
          window_strides
      )
      ops.ReduceWindowWithGeneralPadding(
          operand=make_parameter(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding
      )
      self._ExecuteAndCompareClose(
          c, arguments=(input_array,), expected=[[[5., 7., 9.]]]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowSameUnitStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 1)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.SAME, input_array.shape, window_dimensions,
          window_strides
      )
      ops.ReduceWindowWithGeneralPadding(
          operand=make_parameter(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding
      )
      self._ExecuteAndCompareClose(
          c, arguments=(input_array,), expected=[[[5., 7., 9.], [4., 5., 6.]]]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testReduceWindowValidGeneralStrides(self, dtype):
      if dtype == np.float64 and self.backend.platform == "tpu":
        self.skipTest("TPU doesn't support float64")
      input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 2)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID, input_array.shape, window_dimensions,
          window_strides
      )
      ops.ReduceWindowWithGeneralPadding(
          operand=make_parameter(c, input_array),
          init_value=ops.Constant(c, dtype(0)),
          computation=self._CreateBinaryAddComputation(dtype),
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding
      )
      self._ExecuteAndCompareClose(c, arguments=(input_array,), expected=[[[5., 9.]]])

    @unittest.skipIf(ipu_backend, "`reduce_window` not supported by IPU backend.")
    def testReduceWindowVariadic(self):
      c = self._NewComputation("reducer")
      shape = xla_client.shape_from_pyval(np.array(0, dtype=np.int32))
      shape = shape.with_major_to_minor_layout_if_absent()
      ps = [ops.Parameter(c, i, shape) for i in range(4)]
      which = ops.Ge(ps[0], ps[2])
      ops.Tuple(c, [ops.Select(which, ps[0], ps[2]), ops.Select(which, ps[1], ps[3])])
      reducer = c.build()

      key_array = np.array([[1, 5, 6], [4, 2, 3]], dtype=np.int32)
      val_array = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
      c = self._NewComputation()
      window_dimensions = (2, 1)
      window_strides = (1, 1)
      padding = xla_client.window_padding_type_to_pad_values(
          xla_client.PaddingType.VALID, key_array.shape, window_dimensions,
          window_strides
      )
      ops.ReduceWindowWithGeneralPadding(
          operands=make_parameters(c, [key_array, val_array]),
          init_values=[ops.Constant(c, np.int32(0)),
                       ops.Constant(c, np.int32(0))],
          computation=reducer,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          base_dilations=[],
          window_dilations=[],
          padding=padding
      )
      self._ExecuteAndCompareClose(
          c, arguments=(key_array, val_array), expected=[[[4, 5, 6]], [[10, 8, 9]]]
      )

    @parameterized.named_parameters({
        "testcase_name": "_{}".format(dtype.__name__),
        "dtype": dtype,
    } for dtype in float_dtypes)
    def testWhile(self, dtype):

      def LessThan10Cond():
        c = self._NewComputation("test_lt_10")
        shape = xla_client.shape_from_pyval(np.array(0, dtype=dtype))
        ops.Lt(ops.Parameter(c, 0, shape), ops.Constant(c, dtype(10.)))
        return c.build()

      cond = LessThan10Cond()
      body = self._CreateMulBy2Computation(dtype)
      c = self._NewComputation()
      init_data = dtype(1.)
      ops.While(cond, body, make_parameter(c, init_data))
      self._ExecuteAndCompareClose(c, arguments=(init_data,), expected=[16.])

    def testConditionalTrue(self):
      c = self._NewComputation()
      pred_data = np.bool_(True)
      true_operand = ops.Constant(c, np.float32(3.))
      true_computation = self._CreateMulBy2Computation(np.float32)
      false_operand = ops.Constant(c, np.float32(2.))
      false_computation = self._CreateConstantComputation(np.float32, np.float32)
      ops.Conditional(
          make_parameter(c, pred_data), true_operand, true_computation, false_operand,
          false_computation
      )
      self._ExecuteAndCompareClose(c, arguments=(pred_data,), expected=[6.])

    def testConditionalFalse(self):
      c = self._NewComputation()
      pred_data = np.bool_(False)
      true_operand = ops.Constant(c, np.float32(3.))
      true_computation = self._CreateMulBy2Computation(np.float32)
      false_operand = ops.Constant(c, np.float32(2.))
      false_computation = self._CreateConstantComputation(np.float32, np.float32)
      ops.Conditional(
          make_parameter(c, pred_data), true_operand, true_computation, false_operand,
          false_computation
      )
      self._ExecuteAndCompareClose(c, arguments=(pred_data,), expected=[1.])

    @unittest.skipIf(ipu_backend, "Failing on IPU backend.")
    @unittest.skipIf(cloud_tpu, "not implemented")
    def testInfeedS32Values(self):
      to_infeed = NumpyArrayS32([1, 2, 3, 4])
      c = self._NewComputation()
      ops.GetTupleElement(
          ops.InfeedWithToken(
              ops.CreateToken(c),
              xla_client.shape_from_pyval(to_infeed[0]
                                         ).with_major_to_minor_layout_if_absent()
          ), 0
      )
      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]
      for item in to_infeed:
        device.transfer_to_infeed(item)

      for item in to_infeed:
        result, = xla_client.execute_with_python_values(
            compiled_c, (), backend=self.backend
        )
        self.assertEqual(result, item)

    @unittest.skipIf(ipu_backend, "Not implemented on IPU.")
    @unittest.skipIf(cloud_tpu, "not implemented")
    def testInfeedTuple(self):
      to_infeed = (NumpyArrayS32([1, 2, 3, 4]), NumpyArrayS32([[7], [8]]))
      c = self._NewComputation()
      ops.GetTupleElement(
          ops.InfeedWithToken(
              ops.CreateToken(c),
              xla_client.shape_from_pyval(to_infeed
                                         ).with_major_to_minor_layout_if_absent()
          ), 0
      )
      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]
      device.transfer_to_infeed(to_infeed)

      result = xla_client.execute_with_python_values(
          compiled_c, (), backend=self.backend
      )
      self.assertLen(result, 2)
      np.testing.assert_equal(result[0], to_infeed[0])
      np.testing.assert_equal(result[1], to_infeed[1])

    @unittest.skipIf(ipu_backend, "Failing on IPU backend.")
    @unittest.skipIf(cloud_tpu, "not implemented")
    def testInfeedThenOutfeedS32(self):
      to_round_trip = NumpyArrayS32([1, 2, 3, 4])
      c = self._NewComputation()
      x_and_token = ops.InfeedWithToken(
          ops.CreateToken(c),
          xla_client.shape_from_pyval(to_round_trip[0]
                                     ).with_major_to_minor_layout_if_absent()
      )
      x = ops.GetTupleElement(x_and_token, 0)
      token = ops.GetTupleElement(x_and_token, 1)
      outfeed_shape = xla_client.shape_from_pyval(
          to_round_trip[0]
      ).with_major_to_minor_layout_if_absent()
      ops.OutfeedWithToken(x, token, outfeed_shape)

      compiled_c = self.backend.compile(c.build())
      device = self.backend.local_devices()[0]

      for want in to_round_trip:
        execution = threading.Thread(target=lambda: compiled_c.execute([]))
        execution.start()
        device.transfer_to_infeed(want)
        got = device.transfer_from_outfeed(outfeed_shape)
        execution.join()
        self.assertEqual(want, got)

    def testScatter(self):
      a = np.arange(9).astype(np.int32).reshape((3, 3))
      scatter_indices = np.array([0, 2], dtype=np.int32)
      updates = np.array([[10, 20, 30], [70, 80, 90]], dtype=np.int32)

      dnums = xla_client.ScatterDimensionNumbers()
      dnums.update_window_dims.append(1)
      dnums.inserted_window_dims.append(0)
      dnums.scatter_dims_to_operand_dims.append(0)
      dnums.index_vector_dim = 1

      c = self._NewComputation()
      ops.Scatter(
          *make_parameters(c, [a, scatter_indices, updates]),
          self._CreateBinaryAddComputation(np.int32), dnums
      )
      expected = np.array([[10, 21, 32], [3, 4, 5], [76, 87, 98]], dtype=np.int32)
      self._ExecuteAndCompareClose(
          c, arguments=[a, scatter_indices, updates], expected=[expected]
      )

  tests.append(EmbeddedComputationsTest)

  class DeviceTest(ComputationTest):

    def testPlatform(self):
      for device in self.backend.local_devices():
        self.assertEqual(device.platform, self.backend.platform)

  tests.append(DeviceTest)

  class ErrorTest(ComputationTest):

    def setUp(self):
      super(ErrorTest, self).setUp()
      self.f32_scalar_2 = NumpyArrayF32(2.0)
      self.s32_scalar_2 = NumpyArrayS32(2)
      # self.f32_scalar_2 = NumpyArrayF32([2.0, 3.0, 4.0])
      # self.s32_scalar_2 = NumpyArrayS32([2, 3, 4])

    def testCompileWithWrongElementTypeInLayout(self):
      c = self._NewComputation()
      c.set_op_metadata(xla_client.CurrentSourceInfoMetadata())
      ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
      c.clear_op_metadata()

      options = xla_client.CompileOptions()
      options.argument_layouts = [
          xla_client.Shape.array_shape(np.dtype(np.float32), [])
      ]

      def TestFun():
        return self.backend.compile(c.build(), compile_options=options)

      self.assertRaisesRegex(
          RuntimeError, r".*Invalid argument shape.*"
          r"expected s32\[\], got f32\[\].*", TestFun
      )

    @unittest.skipIf(ipu_backend, "Not implemented on IPU.")
    def testInvokeWithWrongElementType(self):
      c = self._NewComputation()
      c.set_op_metadata(xla_client.CurrentSourceInfoMetadata())
      ops.Parameter(c, 0, xla_client.shape_from_pyval(self.s32_scalar_2))
      c.clear_op_metadata()

      def TestFun():
        return xla_client.execute_with_python_values(
            self.backend.compile(c.build()), [self.f32_scalar_2], self.backend
        )

      # self.assertRaisesRegex(
      #     RuntimeError, r"Invalid argument: Argument does not match.*"
      #     r"want s32\[\], got f32\[\].*", TestFun)
      self.assertRaises(RuntimeError, TestFun)

  tests.append(ErrorTest)

  class ComputationRootTest(ComputationTest):
    """Tests related to setting the root of the computation."""

    def testComputationRootDifferentFromLastOp(self):
      c = self._NewComputation()
      x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
      result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
      ops.Add(result, ops.Constant(c, np.float32(1.618)))

      arg = NumpyArrayF32(1.0)
      compiled_c = self.backend.compile(c.build(result))
      ans, = xla_client.execute_with_python_values(
          compiled_c, [arg], backend=self.backend
      )
      np.testing.assert_allclose(ans, 4.14)

  tests.append(ComputationRootTest)

  class SetShardingTest(ComputationTest):
    """Tests related to set OpSharding."""

    def testSetSharding(self):
      c = self._NewComputation()
      sharding = xla_client.OpSharding()
      sharding.type = xla_client.OpSharding.Type.REPLICATED
      sharding.tile_assignment_dimensions = [1]
      sharding.tile_assignment_devices = [0]
      c.set_sharding(sharding)
      x = ops.Parameter(c, 0, xla_client.shape_from_pyval(NumpyArrayF32(2.0)))
      c.clear_sharding()

      result = ops.Add(x, ops.Constant(c, np.float32(3.14)))
      ops.Add(result, ops.Constant(c, np.float32(1.618)))
      arg = NumpyArrayF32(1.0)
      compiled_c = self.backend.compile(c.build(result))
      ans, = xla_client.execute_with_python_values(
          compiled_c, [arg], backend=self.backend
      )
      np.testing.assert_allclose(ans, 4.14)

  tests.append(SetShardingTest)

  class DeviceAssignmentTest(ComputationTest):

    def testSerialize(self):
      shape = (3, 4)
      device_assignment = xla_client.DeviceAssignment.create(
          np.arange(np.prod(shape)).reshape(*shape)
      )
      self.assertEqual(device_assignment.replica_count(), shape[0])
      self.assertEqual(device_assignment.computation_count(), shape[1])
      serialized = device_assignment.serialize()
      self.assertIsInstance(serialized, bytes)
      self.assertNotEmpty(serialized)

  tests.append(DeviceAssignmentTest)

  class HostCallbackTest(ComputationTest):
    """Tests related to HostCallback."""

    @unittest.skipIf(ipu_backend, "Not implemented on IPU.")
    def testHostCallback(self):

      c = self._NewComputation()
      token = ops.CreateToken(c)

      frontend_attributes = xla_client._xla.FrontendAttributes()
      frontend_attributes["_xla_host_transfer_rendezvous"] = "undef"
      frontend_attributes["_xla_host_transfer_original_type"] = "u32"
      frontend_attributes["_xla_host_transfer_is_lower_bits"] = "false"
      frontend_attributes["_xla_host_transfer_handler_name"] = "undef"
      c.set_frontend_attributes(frontend_attributes)

      send_channel_handle = self.backend.create_channel_handle()
      send_channel_handle.type = (
          xla_client._xla.ChannelHandle_ChannelType.DEVICE_TO_HOST
      )
      send_channel_handle.handle = 1
      ops.SendToHost(
          ops.Constant(c, np.float32(1.25)),
          token,
          shape_with_layout=xla_client.Shape.scalar_shape(np.dtype(np.float32)),
          handle=send_channel_handle
      )

      recv_channel_handle = self.backend.create_channel_handle()
      recv_channel_handle.type = (
          xla_client._xla.ChannelHandle_ChannelType.HOST_TO_DEVICE
      )
      recv_channel_handle.handle = 2
      data = ops.RecvFromHost(
          token,
          shape=xla_client.Shape.scalar_shape(np.dtype(np.float32)),
          handle=recv_channel_handle
      )
      ops.GetTupleElement(data, 0)

      def Identity(x):
        return (x,)

      host_callback = self.backend.make_python_callback_from_host_send_and_recv(
          Identity,
          operand_shapes=[xla_client.Shape.scalar_shape(np.dtype(np.float32))],
          result_shapes=[xla_client.Shape.scalar_shape(np.dtype(np.float32))],
          send_channel_ids=[1],
          recv_channel_ids=[2]
      )

      compiled_c = self.backend.compile(c.build(), host_callbacks=[host_callback])
      c.clear_frontend_attributes()

      results = compiled_c.execute([])
      self.assertLen(results, 1)

      np.testing.assert_equal(results[0].to_py(), np.float32(1.25))

  tests.append(HostCallbackTest)

  class HostCallbackMultiReplicaTest(ComputationTest):
    """Tests related to HostCallback for multi-replica execution."""

    @unittest.skipIf(ipu_backend, "Not implemented on IPU.")
    def testHostCallbackMultiReplica(self):

      c = self._NewComputation()
      token = ops.CreateToken(c)

      frontend_attributes = xla_client._xla.FrontendAttributes()
      frontend_attributes["_xla_host_transfer_rendezvous"] = "undef"
      frontend_attributes["_xla_host_transfer_original_type"] = "u32"
      frontend_attributes["_xla_host_transfer_is_lower_bits"] = "false"
      frontend_attributes["_xla_host_transfer_handler_name"] = "undef"
      c.set_frontend_attributes(frontend_attributes)

      send_channel_handle = self.backend.create_channel_handle()
      send_channel_handle.type = (
          xla_client._xla.ChannelHandle_ChannelType.DEVICE_TO_HOST
      )
      send_channel_handle.handle = 1
      ops.SendToHost(
          ops.ReplicaId(c),
          token,
          shape_with_layout=xla_client.Shape.scalar_shape(np.dtype(np.uint32)),
          handle=send_channel_handle
      )

      recv_channel_handle = self.backend.create_channel_handle()
      recv_channel_handle.type = (
          xla_client._xla.ChannelHandle_ChannelType.HOST_TO_DEVICE
      )
      recv_channel_handle.handle = 2
      data = ops.RecvFromHost(
          token,
          shape=xla_client.Shape.scalar_shape(np.dtype(np.uint32)),
          handle=recv_channel_handle
      )
      ops.GetTupleElement(data, 0)

      def Identity(x):
        return (x,)

      host_callback = self.backend.make_python_callback_from_host_send_and_recv(
          Identity,
          operand_shapes=[xla_client.Shape.scalar_shape(np.dtype(np.uint32))],
          result_shapes=[xla_client.Shape.scalar_shape(np.dtype(np.uint32))],
          send_channel_ids=[1],
          recv_channel_ids=[2]
      )

      num_replicas = 2
      options = xla_client.CompileOptions()
      options.num_replicas = num_replicas
      compiled_c = self.backend.compile(
          c.build(), compile_options=options, host_callbacks=[host_callback]
      )
      c.clear_frontend_attributes()

      results = compiled_c.execute_sharded_on_local_devices([])
      self.assertLen(results, 1)
      self.assertLen(results[0], num_replicas)

      for i in range(num_replicas):
        np.testing.assert_equal(results[0][i].to_py(), np.uint32(i))

  tests.append(HostCallbackMultiReplicaTest)

  return tests


def InstantiateTests(globals_dict, backend_fn, test_prefix="", **kw):
  # Avoid creating a new backend per test (this causes GPU OOM, and is probably
  # inefficient).
  backend_fn = functools.lru_cache(maxsize=None)(backend_fn)
  for klass in TestFactory(backend_fn, **kw):
    test = type(test_prefix + klass.__name__, (klass,), {})
    # Clean up the qualified names of the tests to not include the test factory.
    test.__qualname__ = test.__name__
    globals_dict[test.__name__] = test


backends = {
    "ipu":
        # Using IPU model for quick unit tests.
        lambda: get_ipu_client(
            True,
            IpuPjRtOptions(
                use_ipu_model=True, ipu_model_num_tiles=8, ipu_model_version="ipu2",
                # No dispatch to host by default.
                execute_on_host_flops_limit=-1
            )
        ),
}

if __name__ == "__main__":
  flags.DEFINE_string("backend", "ipu", "Target platform.")
  # pylint: disable=unnecessary-lambda
  InstantiateTests(globals(), lambda: backends[FLAGS.backend]())
  # pylint: enable=unnecessary-lambda
  absltest.main()
