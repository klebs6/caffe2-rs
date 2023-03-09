crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIPoolGradientOp<T,Context> {

    storage:        OperatorStorage,
    context:        Context,

    spatial_scale:  f32,
    pooled_height:  i32,
    pooled_width:   i32,
    order:          StorageOrder,

    /**
      | Input: X, rois, argmaxes, dY (aka
      | "gradOutput")
      |
      | Output: dX (aka "gradInput")
      */
    phantom:        PhantomData<T>,
}

num_inputs!{RoIPoolGradient, 4}

num_outputs!{RoIPoolGradient, 1}

impl<T,Context> RoIPoolGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE_GT(spatial_scale_, 0);
        CAFFE_ENFORCE_GT(pooled_height_, 0);
        CAFFE_ENFORCE_GT(pooled_width_, 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}

register_cpu_operator!{RoIPool, RoIPoolOp<float, CPUContext>}

register_cpu_operator!{RoIPoolGradient, RoIPoolGradientOp<float, CPUContext>}
