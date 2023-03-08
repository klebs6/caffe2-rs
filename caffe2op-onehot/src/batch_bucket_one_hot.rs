crate::ix!();

/**
  | Input is a matrix tensor. Its first dimension
  | is the batch size. For each column, bucketize
  | it based on the boundary values and then
  | do one hot encoding. The `lengths` specifies
  | the number of boundary values for each
  | column. The final number of buckets
  | is this number plus 1. This would also
  | be the expanded feature size. `boundaries`
  | specifies all the boundary values.
  | 
  | -----------
  | @note
  | 
  | each bucket is right-inclusive. That
  | is, given boundary values [b1, b2, b3],
  | the buckets are defined as (-int, b1],
  | (b1, b2], (b2, b3], (b3, inf).
  | 
  | For example
  | 
  | data = [[2, 3], [4, 1], [2, 5]], lengths
  | = [2, 3],
  | 
  | If boundaries = [0.1, 2.5, 1, 3.1, 4.5],
  | then
  | 
  | output = [[0, 1, 0, 0, 1, 0, 0], [0, 0,
  | 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]]
  | 
  | If boundaries = [0.1, 2.5, 1, 1, 3.1],
  | then
  | 
  | output = [[0, 1, 0, 0, 0, 1, 0], [0, 0,
  | 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1]]
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchBucketOneHotOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BatchBucketOneHot, 3}

num_outputs!{BatchBucketOneHot, 1}

inputs!{BatchBucketOneHot, 
    0 => ("data", "input tensor matrix"),
    1 => ("lengths", "the size is the same as the width of the `data`"),
    2 => ("boundaries", "bucket boundaries")
}

outputs!{BatchBucketOneHot, 
    0 => ("output", "output matrix that expands each input column with one hot encoding based on the bucketization")
}

tensor_inference_function!{BatchBucketOneHot, /* (TensorInferenceForBucketBatchOneHot) */}

disallow_input_fillers!{BatchBucketOneHot}

input_tags!{
    BatchBucketOneHotOp {
        X,
        Lens,
        Boundaries
    }
}

output_tags!{
    BatchBucketOneHotOp {
        OneHot
    }
}

impl<Context> BatchBucketOneHotOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
