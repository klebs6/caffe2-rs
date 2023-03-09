crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIAlignRotatedGradientOp<T,Context> {

    storage:         OperatorStorage,
    context:         Context,

    spatial_scale:   f32,
    pooled_height:   i32,
    pooled_width:    i32,
    sampling_ratio:  i32,
    aligned:         bool,

    /**
      | Input: X, rois, dY (aka "gradOutput");
      | 
      | Output: dX (aka "gradInput")
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{RoIAlignRotatedGradient, 3}

num_outputs!{RoIAlignRotatedGradient, 1}

inputs!{RoIAlignRotatedGradient, 
    0 => ("X",    "See RoIAlignRotated."),
    1 => ("RoIs", "See RoIAlignRotated."),
    2 => ("dY",   "Gradient of forward output 0 (Y)")
}

outputs!{RoIAlignRotatedGradient, 
    0 => ("dX",   "Gradient of forward input 0 (X)")
}

impl<T,Context> RoIAlignRotatedGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            sampling_ratio_( this->template GetSingleArgument<int>("sampling_ratio", -1)),
            aligned_(this->template GetSingleArgument<bool>("aligned", false)) 
        DCHECK_GT(spatial_scale_, 0);
        DCHECK_GT(pooled_height_, 0);
        DCHECK_GT(pooled_width_, 0);
        DCHECK_GE(sampling_ratio_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}
