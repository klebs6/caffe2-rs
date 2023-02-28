crate::ix!();

pub struct ResizeNearest3DGradientOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:         OperatorStorage,
    context:         Context,

    temporal_scale:  T,
    height_scale:    T,
    width_scale:     T,
    order:           StorageOrder,

    /*
      | Input: dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{ResizeNearest3DGradient, 2}

num_outputs!{ResizeNearest3DGradient, 1}

args!{ResizeNearest3DGradient, 
    0 => ("temporal_scale", "Scale along temporal dimension"),
    1 => ("width_scale", "Scale along width dimension"),
    2 => ("height_scale", "Scale along height dimension")
}

impl<T,Context> ResizeNearest3DGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            temporal_scale_(1),
            height_scale_(1),
            width_scale_(1),
            order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        temporal_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("temporal_scale", 1));
        height_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("height_scale", 1));
        width_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("width_scale", 1));

        CAFFE_ENFORCE_GT(temporal_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        CAFFE_ENFORCE_GT(width_scale_, 0);

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
}
