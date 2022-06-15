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
import threading
import unittest

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import make_ipu_client


ops = xla_client.ops
FLAGS = flags.FLAGS

# We choose to ignore pylint's complaints about complex comprehensions, which we
# use widely for parameterizing tests.
# pylint: disable=g-complex-comprehension


def make_parameter(c, arr, idx = 0):
  """Make ops parameter, necessary for IPU testing. 
  """
  # We need to modify slightly the official XLA client unit tests:
  # 1. Poplar Compiler will Skip engine compilation if output is constant(compute by CPU)， So test as Parameter input rather than constant
  # 2. Parameter's shape with layout will cause an check error when IPU, so call with_major_to_minor_layout_if_absent to ge a default layout
  return ops.Parameter(c, idx, xla_client.shape_from_pyval(arr).with_major_to_minor_layout_if_absent())

def make_parameters(c, arrs):
  return [make_parameter(c, arr, idx) for idx, arr in enumerate(arrs)]


def TestFactory(xla_backend):
  tests = []
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
          compiled_c, arguments, backend=self.backend)

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
      self._ExecuteAndAssertWith(np.testing.assert_equal, c, arguments,
                                 expected)

    def _ExecuteAndCompareClose(self,
                                c,
                                arguments=(),
                                expected=None,
                                rtol=1e-4,
                                atol=0):
      self._ExecuteAndAssertWith(
          functools.partial(np.testing.assert_allclose, rtol=rtol, atol=atol),
          c, arguments, expected)

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
          c, arguments=args, 
          expected=[np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)])

    # pyformat: disable
    @parameterized.named_parameters({
        "testcase_name": "_{}_{}".format(src_dtype.__name__,
                                         dst_dtype.__name__),
        "src_dtype": src_dtype,
        "dst_dtype": dst_dtype,
    } for src_dtype, dst_dtype in itertools.permutations(
        [np.bool_, np.int32, np.float32], 2))
    # pyformat: enable
    def testConvertElementType(self, src_dtype, dst_dtype):
      if ((src_dtype in [np.int64, np.float64] or
           dst_dtype in [np.int64, np.float64]) and
          self.backend.platform == "tpu"):
        self.skipTest("TPU doesn't support float64")
      c = self._NewComputation()
      x = np.array([0, 1, 0, 0, 1], dtype=src_dtype)
      ops.ConvertElementType(
          make_parameter(c, x), xla_client.dtype_to_etype(dst_dtype))
      
      result = xla_client.execute_with_python_values(
          self.backend.compile(c.build()), (x,), backend=self.backend)
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
    "ipu": make_ipu_client,
}

if __name__ == "__main__":
  flags.DEFINE_string("backend", "ipu", "Target platform.")
  # pylint: disable=unnecessary-lambda
  InstantiateTests(globals(), lambda: backends[FLAGS.backend]())
  # pylint: enable=unnecessary-lambda
  absltest.main()
