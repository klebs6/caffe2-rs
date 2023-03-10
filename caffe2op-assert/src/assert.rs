crate::ix!();

/**
 | Takes in a tensor of type *bool*, *int*, *long*,
 | or *long long* and checks if all values are True
 | when coerced into a boolean. In other words, for
 | non-bool types this asserts that all values in the
 | tensor are non-zero. 
 |
 | If a value is False after coerced into a boolean,
 | the operator throws an error. 
 |
 | Else, if all values are True, nothing is
 | returned. For tracability, a custom error message
 | can be set using the `error_msg` argument.
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc
 |
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AssertOp<Context> {
    context:    Context,
    cmp_tensor: Tensor, //{CPU};
    error_msg:  String,
}

register_cpu_operator!{
    Assert, 
    AssertOp<CPUContext>
}

num_inputs!{Assert, 1}

num_outputs!{Assert, 0}

inputs!{Assert, 
    0 => ("X", "(*Tensor*): input tensor")
}

args!{Assert, 
    0 => ("error_msg", "(*string*): custom error message to be thrown when the input does not pass assertion")
}
