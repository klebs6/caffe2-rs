crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SoftmaxWithLossGradientOp<T,Context> {
    storage:                OperatorStorage,
    context:                Context,
    scale:                  f32,
    label_prob_mode:        i32,
    average_by_batch_size:  i32,

    /// not used?
    sum_multiplier:         Tensor, /// {Context::GetDeviceType()};

    /// unignored weights
    weights:                Tensor,
    total_weight_ptr:       Tensor,
    order:                  StorageOrder,
    only_loss:              bool,
    axis:                   i32,

    scratch:                Tensor, // {Context::GetDeviceType()};

    /**
      | Input: X, T, P, dY;
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

register_cpu_operator!{SoftmaxWithLossGradient, SoftmaxWithLossGradientOp<f32, CPUContext>}

num_outputs!{SoftmaxWithLossGradient, 1}

impl<T,Context> SoftmaxWithLossGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            label_prob_mode_( this->template GetSingleArgument<int>("label_prob", 0)),
            average_by_batch_size_( this->template GetSingleArgument<int>("average_by_batch_size", 0)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            only_loss_(this->template GetSingleArgument<bool>("only_loss", false)),
            axis_(this->template GetSingleArgument<int>("axis", 1)) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}
