/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GML_ST_EXTENSION_OPS
#define GML_ST_EXTENSION_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"
include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td"

def TensorOrMemref : AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;

def GMLST_DynamicBroadcastInDimOp : GMLST_Op<"dynamic_broadcast_in_dim",
    [NoSideEffect,
     TypesMatchWith<"result and init types match", "init", "result", "$_self">,
     DeclareOpInterfaceMethods<FusionInterface>]> {
  let summary = [{Destination-style twin for `mhlo.dynamic_broadcast_in_dim`}];

  let arguments = (ins
    // Input args
    TensorOrMemref:$operand,
    // Output arg
    TensorOrMemref:$init,

    DenseI64ArrayAttr:$broadcast_dimensions,
    OptionalAttr<DenseI64ArrayAttr>:$known_expanding_dimensions,
    OptionalAttr<DenseI64ArrayAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    `ins` `(` $operand `:` type($operand) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict
  }];
}

def GMLST_GatherOp : GMLST_Op<"gather", [
  NoSideEffect,
  TypesMatchWith<"result and init types match", "init", "result", "$_self">
]> {
  let summary = "Destination-style twin for `mhlo.gather`";
  let description = [{
    Does not currently support the full interface of mhlo.gather. In particular:
    - index_vector_dim is start_indices.shape.rank - 1
    - slice_sizes is [1,1,...]
    - offset_dims is []
    - collapsed_slice_dims is range(operand.shape.rank)
    - start_index_map is range(slice_sizes.shape[index_vector_dim])
  }];
  let arguments = (ins
    // Input args
    TensorOrMemref:$operand,
    I64Tensor:$start_indices,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    `ins` `(` $operand `:` type($operand) `,`
              $start_indices `:` type($start_indices) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict
  }];
}

def GMLST_ScatterOp : GMLST_Op<"scatter", [
    TypesMatchWith<"result and init types match", "init", "result", "$_self">,
    NoSideEffect]> {
  let summary = "Destination-style twin for `mhlo.scatter`";
  let description = [{
    Caveats:
    - the variadic case is not supported.
    - update_computation is sum.
    - Only point updates are supported
      - update_window_dims is []
      - inserted_window_dims is range(operand.shape.rank)
      - scatter_dims_to_operand_dims is range(indices.shape.rank)
      - index_vector_dim is indices.shape.rank-1
  }];
  let arguments = (ins
    // Input args
    TensorOf<[I32, I64]>:$indices,
    TensorOrMemref:$updates,
    // Output arg
    TensorOrMemref:$init
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    `ins` `(` $indices `:` type($indices) `,`
              $updates `:` type($updates) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict
  }];
}

#endif // GML_ST_EXTENSION_OPS
