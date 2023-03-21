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

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xla_extension
from tensorflow.compiler.plugin.poplar.xla_client.python.ipu_xla_client import (
    IpuDeviceMeshManager, IpuPjRtOptions, get_ipu_client, IpuPjRtClientState,
    IpuPjRtExecutableRunInfo
)

ops = xla_client.ops

# Skipping some tests if no local IPU hardware.
ipu_hw_available = IpuDeviceMeshManager.has_local_ipu_hardware()

try:
  # TODO: fix error when no IPU hardware available.
  # TODO: support non-global backend?
  ipu_backend = get_ipu_client(True, IpuPjRtOptions(use_ipu_model=False))
except RuntimeError:
  ipu_backend = None


class IpuPjrtClientStateTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.backend = ipu_backend

  def setUp(self):
    super(IpuPjrtClientStateTest, self).setUp()
    if not ipu_hw_available:
      self.skipTest("No IPU hardware available, skipping test.")
    # IPU mesh manager for state managment.
    self.mesh_manager = self.backend.local_devices()[0].ipu_mesh_manager

  def testIpuPjRtclient__executable_run_info__init(self):
    info = IpuPjRtExecutableRunInfo(mesh_id=1, executable_id=2, run_id=3)
    self.assertEqual(info.mesh_id, 1)
    self.assertEqual(info.executable_id, 2)
    self.assertEqual(info.run_id, 3)

  def testIpuPjRtclient__client_state__initialize(self):
    state = IpuPjRtClientState.initialize(self.mesh_manager)
    self.assertIsInstance(state, IpuPjRtClientState)
    self.assertEqual(len(state), self.mesh_manager.count(1))
    self.assertEqual([m.mesh_id for m in state.active_meshes], list(range(len(state))))
    self.assertEqual([m.executable_id for m in state.active_meshes], [0] * len(state))
    self.assertEqual([m.run_id for m in state.active_meshes], [0] * len(state))

  def testIpuPjRtclient__client_state__update__same_mesh_topology(self):
    state = IpuPjRtClientState.initialize(self.mesh_manager)
    run_info = IpuPjRtExecutableRunInfo(mesh_id=1, executable_id=2, run_id=3)
    state_updated = state.update(run_info, self.mesh_manager)
    self.assertEqual(len(state_updated), len(state))
    # New updated mesh info.
    self.assertEqual(state_updated.active_meshes[1].mesh_id, run_info.mesh_id)
    self.assertEqual(
        state_updated.active_meshes[1].executable_id, run_info.executable_id
    )
    self.assertEqual(state_updated.active_meshes[1].run_id, run_info.run_id)
    for idx in range(len(state)):
      if idx == run_info.mesh_id:
        continue
      # All active meshes.
      self.assertTrue(
          state_updated.is_active_mesh(state_updated.active_meshes[idx].mesh_id)
      )
      # Other meshes are the same.
      self.assertEqual(
          state_updated.active_meshes[idx].mesh_id, state.active_meshes[idx].mesh_id
      )
      self.assertEqual(
          state_updated.active_meshes[idx].executable_id,
          state.active_meshes[idx].executable_id
      )
      self.assertEqual(
          state_updated.active_meshes[idx].run_id, state.active_meshes[idx].run_id
      )
    # Last mesh should not be active.
    self.assertTrue(
        state_updated.is_active_mesh(state_updated.active_meshes[-1].mesh_id)
    )

  def testIpuPjRtclient__client_state__update__different_topology(self):
    state = IpuPjRtClientState.initialize(self.mesh_manager)
    mesh = self.mesh_manager.find([0, 1])
    run_info = IpuPjRtExecutableRunInfo(mesh_id=mesh.id, executable_id=2, run_id=3)
    state_updated = state.update(run_info, self.mesh_manager)
    self.assertEqual(len(state_updated), len(state) - 1)
    # New updated mesh info: last one normally.
    self.assertEqual(state_updated.active_meshes[-1].mesh_id, run_info.mesh_id)
    self.assertEqual(
        state_updated.active_meshes[-1].executable_id, run_info.executable_id
    )
    self.assertEqual(state_updated.active_meshes[-1].run_id, run_info.run_id)
    # All other meshes should have greater id.
    for m in state_updated.active_meshes:
      self.assertGreaterEqual(m.mesh_id, 2)


def make_compile_options(devices) -> xla_client.CompileOptions:
  """Create compile options, from list (of list) of device ids.
  """
  device_assignment = xla_extension.DeviceAssignment.create(devices)
  opts = xla_client.CompileOptions()
  opts.device_assignment = device_assignment
  opts.num_partitions = device_assignment.computation_count()
  opts.num_replicas = device_assignment.replica_count()
  opts.parameter_is_tupled_arguments = False
  opts.tuple_arguments = False
  return opts


def make_reduce_xla_computation(
    devices, reduce_op: xla_client.XlaOp
) -> xla_client.XlaComputation:
  """Build a simple Xla computation reduce op.
  """
  devices = np.ravel(devices)
  input_aval = np.array(0.0, dtype=np.float32)
  replica_group = xla_client.ReplicaGroup()
  replica_group.replica_ids = list(devices)

  reduce_builder = xla_client.XlaBuilder("reduce_op")
  p0 = ops.Parameter(reduce_builder, 0, xla_client.shape_from_pyval(input_aval))
  p1 = ops.Parameter(reduce_builder, 1, xla_client.shape_from_pyval(input_aval))
  reduce_op(p0, p1)
  return reduce_builder.build(), replica_group


class IpuPjrtClientExecutableTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.backend = ipu_backend

  def setUp(self):
    super(IpuPjrtClientExecutableTest, self).setUp()
    if not ipu_hw_available:
      self.skipTest("No IPU hardware available, skipping test.")
    # IPU mesh manager for state managment.
    self.mesh_manager = self.backend.local_devices()[0].ipu_mesh_manager

  @parameterized.parameters([
      [1, [0]],
      [2, [0, 1]],
      [4, [0, 1, 2, 3]],
  ])
  def testIpuPjRtclientHw__get_default_device_assignment__proper_device_assignment(
      self, num_replicas, expected_ids
  ):
    device_assignment = self.backend.get_default_device_assignment(num_replicas, 1)
    self.assertEqual(len(device_assignment), num_replicas)
    self.assertEqual([d[0].id for d in device_assignment], expected_ids)

  @parameterized.parameters([
      [3, 1],
      [1, 2],
      [128, 1],
  ])
  def testIpuPjRtclientHw__get_default_device_assignment__unsupported_mesh_size(
      self, num_replicas, num_partitions
  ):
    with self.assertRaises(xla_extension.XlaRuntimeError):
      self.backend.get_default_device_assignment(num_replicas, num_partitions)


class IpuPjrtClientExecutableHardwareTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.backend = ipu_backend

  def setUp(self):
    super(IpuPjrtClientExecutableHardwareTest, self).setUp()
    if not ipu_hw_available:
      self.skipTest("No IPU hardware available, skipping test.")
    # IPU mesh manager for state managment.
    self.mesh_manager = self.backend.local_devices()[0].ipu_mesh_manager

  def testIpuPjRtclientHw__executable__single_ipu__successful_multi_runs(self):
    compile_opts = make_compile_options([[0]])
    c = xla_client.XlaBuilder(self.id())

    arg0 = np.array([10, 15, -2, 7], dtype=np.float32)
    arg1 = np.array([1, 3, -7, 9], dtype=np.float32)
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    p1 = ops.Parameter(c, 1, xla_client.shape_from_pyval(arg1))
    ops.Mul(p0, p1)
    executable = self.backend.compile(c.build(), compile_opts)

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

  def testIpuPjRtclientHw__executable_multi_ipus__successful_compilation(self):
    # NOTE: good to test different IPU meshes in tests, to catch weird bugs!
    device_ids = [[0], [1]]
    compile_opts = make_compile_options(device_ids)
    arg0 = np.array([10, -2, 7], dtype=np.float32)

    c = xla_client.XlaBuilder(self.id())
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    reduce_op, replica_group = make_reduce_xla_computation(device_ids, ops.Add)
    ops.AllReduce(p0, reduce_op, replica_groups=[replica_group])
    executable = self.backend.compile(c.build(), compile_opts)

    # TODO: additional testing on PjRt executable.
    npt.assert_array_equal([d.id for d in executable.local_devices()],
                           np.ravel(device_ids))

  def testIpuPjRtclientHw__executable_multi_ipus__all_reduce_sum(self):
    # IPU POD4 all reduce
    # NOTE: good to test different IPU meshes in tests, to catch weird bugs!
    device_ids = [[0], [1], [2], [3]]
    num_devices = np.array(device_ids).size
    compile_opts = make_compile_options(device_ids)
    arg0 = np.array([10, -2, 1.5], dtype=np.float32)

    c = xla_client.XlaBuilder(self.id())
    p0 = ops.Parameter(c, 0, xla_client.shape_from_pyval(arg0))
    reduce_op, replica_group = make_reduce_xla_computation(device_ids, ops.Add)
    ops.AllReduce(p0, reduce_op, replica_groups=[replica_group])
    executable = self.backend.compile(c.build(), compile_opts)

    # Different data on every IPU.
    inputs = [[arg0 * (idx + 1)] for idx in range(num_devices)]
    outputs = xla_client.execute_with_python_values_replicated(
        executable, inputs, self.backend
    )

    expected_output = np.sum([v[0] for v in inputs], axis=0)
    self.assertEqual(len(outputs), num_devices)
    # TODO: additional testing on output device assignment.
    for out in outputs:
      npt.assert_array_almost_equal(out[0], expected_output)


if __name__ == "__main__":
  absltest.main()
