crate::ix!();

/**
  | Retrieve the argmax of an axis dimension
  | specified by the `axis` argument.
  | 
  | Given an input tensor and two arguments
  | (`axis` and `keepdims`), returns a
  | tensor containing the indices of the
  | largest element along the given axis.
  | 
  | If the `keepdims` arg is *True* (default),
  | the shape of the output tensor matches
  | the input tensor except the `axis` dimension
  | equals 1.
  | 
  | Else, the `axis` dimension of the output
  | tensor is removed.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc
  |
  */
pub struct ArgMaxReducer<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{ArgMax, 1}

num_outputs!{ArgMax, 1}

inputs!{ArgMax, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{ArgMax, 
    0 => ("Indices", "*(type: Tensor`<float>`)* Tensor of indices for the largest values.")
}

args!{ArgMax, 
    0 => ("axis", "*(type: int; default: -1)* The axis to get argmax."),
    1 => ("keepdims", "*(type: bool; default: True)* If True (default), 
        the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. 
        Else, the `axis` dimension of the output tensor is removed.")
}

tensor_inference_function!{
    ArgMax, 
    InferTensor
}
