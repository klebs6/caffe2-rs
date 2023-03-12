crate::ix!();

/**
  | This operator computes the cross entropy
  | between a $NxD$ dimensional input data
  | tensor $X$ and a one dimensional input
  | label tensor $label$. The op produces
  | a single length $N$ output tensor $Y$.
  | Here, $N$ is considered the batch size
  | and $D$ is the size of each element in
  | the batch. In practice, it is most commonly
  | used at the end of models as a part of the
  | loss computation, after the
  | 
  | SoftMax operator and before the AveragedLoss
  | operator. The cross entropy operation
  | is defined as follows
  | 
  | $$Y_i = -log(X_{ij})$$
  | 
  | where ($i$, $j$) is the classifier's
  | prediction of the $j$th class (the correct
  | one), and $i$ is the batch size. Each
  | log has a lower limit for numerical stability.
  | 
  | The difference between *LabelCrossEntropy*
  | and *CrossEntropy* is how the labels
  | are specified.
  | 
  | Here, the labels are a length $N$ list
  | of integers, whereas in CrossEntropy
  | the labels are a $NxD$ dimensional matrix
  | of one hot label vectors. However, the
  | results of computation should be the
  | same, as shown in the two examples where
  | ($i$, $j$) is the classifier's prediction
  | of the $j$th class (the correct one),
  | and $i$ is the batch size. Each log has
  | a lower limit for numerical stability.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LabelCrossEntropyOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    // Input: X, label
    //
    // Output: Y
    //
    phantom: PhantomData<T>,
}

num_inputs!{LabelCrossEntropy, 2}

num_outputs!{LabelCrossEntropy, 1}

inputs!{LabelCrossEntropy, 
    0 => ("X", "Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes."),
    1 => ("label", "Blob containing the labels used to compare the input. $label$ is a length $N$ list of integers, where each element is the integer label for the $n$th element of the batch.")
}

outputs!{LabelCrossEntropy, 
    0 => ("Y", "Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.")
}

identical_type_and_shape_of_input_dim!{LabelCrossEntropy, (0, 0)}

impl<T,Context> LabelCrossEntropyOp<T, Context> {

    pub const fn k_log_threshold() -> T {
        todo!();
        //return static_cast<T>(1e-20);
    }
}
