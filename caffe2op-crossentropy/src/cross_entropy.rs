crate::ix!();

/**
  | This operator computes the cross entropy
  | between a $NxD$ dimensional input data
  | tensor $X$ and a $NxD$ dimensional input
  | label tensor $label$.
  | 
  | The op produces a single length $N$ output
  | tensor $Y$. Here, $N$ is considered
  | the batch size and $D$ is the size of each
  | element in the batch. In practice, it
  | is most commonly used at the end of models
  | as a part of the loss computation, after
  | the SoftMax operator and before the
  | AveragedLoss operator. The cross entropy
  | operation is defined as follows
  | 
  | $$Y_i = \sum_j (label_{ij} * log(X_{ij}))$$
  | 
  | where ($i$, $j$) is the classifier's
  | prediction of the $j$th class (the correct
  | one), and $i$ is the batch size. Each
  | log has a lower limit for numerical stability.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CrossEntropyOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, label
      | 
      | Output: Y
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{CrossEntropy, 2}

num_outputs!{CrossEntropy, 1}

inputs!{CrossEntropy, 
    0 => ("X",     "Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes."),
    1 => ("label", "Blob containing the labels used to compare the input. $label$ is the same shape as $X$.")
}

outputs!{CrossEntropy, 
    0 => ("Y", "Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.")
}

identical_type_and_shape_of_input_dim!{CrossEntropy, (0, 0)}

impl<T, Context> CrossEntropyOp<T, Context> {

    pub const fn k_log_threshold() -> T {
        todo!();
        //return static_cast<T>(1e-20);
    }
}
