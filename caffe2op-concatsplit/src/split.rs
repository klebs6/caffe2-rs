crate::ix!();

pub const kSplitOpInputSize: i32 = 2;

/**
  | Split an `input` tensor into a list of
  | tensors, along the axis specified by
  | the `axis` dimension. The lengths of
  | the split can be specified using argument
  | `split` or optional second input blob
  | to the operator. Otherwise, the tensor
  | is split to equal sized parts.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SplitOp<Context> {
    context: Context,

    axis:     i32,
    add_axis: i32,
    split:    Vec<i32>,

    /*
      | Input: X, optionally split
      | The split tensor is stored in CPU.
      |
      */
}

num_inputs!{Split, (1,2)}

num_outputs!{Split, (1,INT_MAX)}

inputs!{Split, 
    0 => ("input", "(*Tensor*): tensor to split"),
    1 => ("split", "(*Tensor`<int>`*): [OPTIONAL] list of output lengths (see also arg `split`)")
}

outputs!{Split, 
    0 => ("[output_0, output_1, ...]", "(*Tensor*): output tensor")
}

args!{Split, 
    0 => ("axis", "(*int*): axis to split on"),
    1 => ("add_axis", "*(type: int)* Pass non-zero integer to remove the axis specified in `axis` to all input tensors."),
    2 => ("split", "(*Tuple(int)*): length of each output"),
    3 => ("order", "(*string*): order of dimensions of input and output blobs; either NCHW or NHWC")
}

inherit_onnx_schema!{Split}

tensor_inference_function!{Split, TensorInferenceForSplit}

device_inference_function!{Split, splitOpDevInfer}

