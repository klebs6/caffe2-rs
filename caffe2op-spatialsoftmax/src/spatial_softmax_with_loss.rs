crate::ix!();

/**
  | Combined Spatial Softmax and Cross-Entropy
  | loss operator.
  | 
  | Similar to SoftmaxWithLoss, this operator
  | computes the spatial softmax normalized
  | values for each layer in the batch of
  | the given input, after which cross-entropy
  | loss is computed.
  | 
  | This operator is numerically more stable
  | than separate Softmax and CrossEntropy
  | ops.
  | 
  | The inputs are a 2-D tensor (Tensor)
  | of size (batch_size x input_feature_dimensions)
  | and tensor of labels (ground truth).
  | 
  | Output is tensor with the probability
  | for each label in a pixel for each example
  | (N x D x W x H) and averaged loss (scalar).
  | 
  | For spatial softmax, weighting is by
  | x,y position of the input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialSoftmaxWithLossOp<T,Context> {

    storage:           OperatorStorage,
    context:           Context,

    scale:             f32,
    order:             StorageOrder,

    /// Per example loss
    losses:            Tensor,

    /// per example row max
    rowmax:            Tensor,

    /// unignored weights
    weights:           Tensor,

    /// Vector of ones for summing via dot prod
    sum_multiplier:    Tensor,

    total_weight_ptr:  Tensor,

    /// {Context::GetDeviceType()};
    scratch:           Tensor,

    /**
      | Input: X (logits), T (labels);
      | 
      | Output: P (probs), Y
      |
      */
    phantom: PhantomData<T>,
}

impl<T,Context> SpatialSoftmaxWithLossOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}
