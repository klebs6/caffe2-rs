crate::ix!();

/**
  | The *OneHot* op accepts two inputs *indices*
  | and *index_size_tensor*, and produces
  | a single output one_hots*. For each
  | index in *indices* the op creates a one-hot
  | row in *one_hots* of length index_size_tensor*
  | where all entries are zero except the
  | entry at the index is 1. The size of one_hots*
  | is *len(indices)* x *index_size_tensor*.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct OneHotOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{OneHot, 2}

num_outputs!{OneHot, 1}

inputs!{OneHot, 
    0 => ("indices", "The active index for each example in the batch."),
    1 => ("index_size_tensor", "Scalar with the size of the index. Must be in CPU context")
}

outputs!{OneHot, 
    0 => ("one_hots", "Matrix of size len(indices) x index_size")
}

// TODO: enable the filler
disallow_input_fillers!{OneHot}
