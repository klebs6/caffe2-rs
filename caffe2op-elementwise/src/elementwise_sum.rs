crate::ix!();

/**
 | Element-wise sum of each of the input tensors. The
 | first input tensor can be used in-place as the
 | output tensor, in which case the sum will be done
 | in place and results will be accumulated the first
 | input tensor. All inputs and outputs must have the
 | same shape and data type.
 |
 | Github Links:
 |
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc
 |
 */
pub struct ElementwiseSumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    Sum, 
    ElementwiseSumOp<CPUContext>
}

num_inputs!{Sum, (1,INT_MAX)}

num_outputs!{Sum, 1}

inputs!{Sum, 
    0 => ("A", "*(type: Tensor`<float>`)* First tensor to be added element-wise."),
    1 => ("B", "*(type: Tensor`<float>`)* Second tensor to be added element-wise.")
}

outputs!{Sum, 
    0 => ("C", "*(type: Tensor`<float>`)* Sum of A and B.")
}

allow_inplace!{Sum, vec![(0, 0)]}

cost_inference_function!{Sum, CostInferenceForSum}

inputs_can_cross_devices!{Sum}

identical_type_and_shape_of_input!{Sum, 0}

inherit_onnx_schema!{Sum}
