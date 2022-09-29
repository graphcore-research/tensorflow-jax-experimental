# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""
Operations and utilities related to the Graphcore IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=wildcard-import,unused-import
# from tensorflow.python.ipu import data
from tensorflow.python.ipu import config
# from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import ipu_compiler
# from tensorflow.python.ipu import ipu_infeed_queue
# from tensorflow.python.ipu import ipu_multi_worker_strategy
# from tensorflow.python.ipu import ipu_outfeed_queue
# from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
# from tensorflow.python.ipu import serving
from tensorflow.python.ipu import sharding
# from tensorflow.python.ipu import utils
# from tensorflow.python.ipu import vertex_edsl

# Lazy load estimator API to prevent dependency problems with Keras.
from tensorflow.python.util import lazy_loader
ipu_run_config = lazy_loader.LazyLoader(
    "ipu_run_config", globals(), "tensorflow.python.ipu.ipu_run_config")
ipu_session_run_hooks = lazy_loader.LazyLoader(
    "ipu_session_run_hooks", globals(),
    "tensorflow.python.ipu.ipu_session_run_hooks")
ipu_estimator = lazy_loader.LazyLoader("ipu_estimator", globals(),
                                       "tensorflow.python.ipu.ipu_estimator")
ipu_pipeline_estimator = lazy_loader.LazyLoader(
    "ipu_pipeline_estimator", globals(),
    "tensorflow.python.ipu.ipu_pipeline_estimator")

# pylint: enable=wildcard-import,unused-import

sharding.enable_sharded_gradient_tape()
