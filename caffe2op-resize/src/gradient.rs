crate::ix!();

pub struct ResizeNearestGradientOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    width_scale:   T,
    height_scale:  T,
    order:         StorageOrder,

    /*
      | Input: dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{ResizeNearestGradient, (2,3)}

num_outputs!{ResizeNearestGradient, 1}

args!{ResizeNearestGradient, 
    0 => ("width_scale", "Scale along width dimension"),
    1 => ("height_scale", "Scale along height dimension")
}

impl<T,Context> ResizeNearestGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            width_scale_(1),
            height_scale_(1),
            order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        width_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("width_scale", 1));
        height_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("height_scale", 1));

        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
}

register_gradient!{ResizeNearest, GetResizeNearestGradient}
