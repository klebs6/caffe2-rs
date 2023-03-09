crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UpsampleBilinearGradientOp<T, Context> {
    storage:      OperatorStorage,
    context:      Context,
    width_scale:  T,
    height_scale: T,

    /*
      | Input: dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{UpsampleBilinearGradient, (2,3)}

num_outputs!{UpsampleBilinearGradient, 1}

args!{UpsampleBilinearGradient, 
    0 => ("width_scale",  "Scale along width dimension"),
    1 => ("height_scale", "Scale along height dimension")
}

impl<T,Context> UpsampleBilinearGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            width_scale_(1),
            height_scale_(1) 

        width_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("width_scale", 1));
        height_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("height_scale", 1));
        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        */
    }
}
