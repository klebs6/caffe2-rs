crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialSoftmaxWithLossGradientOp<T,Context> {
    storage:           OperatorStorage,
    context:           Context,
    scale:             f32,
    sum_multiplier:    Tensor,

    /// unignored weights
    weights:           Tensor,
    total_weight_ptr:  Tensor,
    order:             StorageOrder,
    only_loss:         bool,

    /// {Context::GetDeviceType()};
    scratch:           Tensor,

    /**
      | Input: X, T, P, dY;
      | 
      | Output: dX
      |
      */
    phantom:           PhantomData<T>,
}

num_outputs!{SpatialSoftmaxWithLossGradient, 1}

impl<T,Context> SpatialSoftmaxWithLossGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            only_loss_(this->template GetSingleArgument<bool>("only_loss", false)) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}
