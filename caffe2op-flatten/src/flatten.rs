crate::ix!();

/**
 | Flattens the input tensor into a 2D matrix. If input tensor has shape
 | $(d_0, d_1, ..., d_n)$ then the output will have shape
 | $\bigl((d_0 * d_1 * ... * d_{(axis-1)}), (d_{axis} * d_{(axis+1)} * ... * d_n)\bigr)$.
 |
 | Github Links:
 |
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FlattenOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axis:    i32,
}

num_inputs!{Flatten, 1}

num_outputs!{Flatten, 1}

inputs!{Flatten, 
    0 => ("X", "*(type: Tensor)* Input Tensor of rank >= axis.")
}

outputs!{Flatten, 
    0 => ("Y", "*(type: Tensor)* A 2D tensor with the contents of the input tensor, with input dimensions up to `axis` flattened to the outer dimension of the output and the remaining input dimensions flattened into the inner dimension of the output.")
}

args!{Flatten, 
    0 => ("axis", "*(type: int; default: 1)* Indicates up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.")
}

tensor_inference_function!{Flatten, TensorInferenceForFlatten}

inherit_onnx_schema!{Flatten}

register_cpu_operator!{Flatten, FlattenOp<CPUContext>}
