/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("IpuHardSigmoid")
    .Input("features: dtype")
    .Output("activations: dtype")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IpuHardSigmoidGrad")
    .Input("gradients: dtype")
    .Input("features: dtype")
    .Output("backprops: dtype")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

}  // namespace tensorflow
