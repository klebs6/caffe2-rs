crate::ix!();

declare_export_caffe2_op_to_c10![RoIAlignRotated];

/**
  | Similar to RoIAlign but can handle rotated
  | region proposals.
  | 
  | Based on https://arxiv.org/abs/1703.01086.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIAlignRotatedOp<T,Context> {

    storage:         OperatorStorage,
    context:         Context,

    order:           StorageOrder,
    spatial_scale:   f32,
    pooled_height:   i32,
    pooled_width:    i32,
    sampling_ratio:  i32,
    aligned:         bool,

    /**
      | Input: X, rois;
      | 
      | Output: Y
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{RoIAlignRotated, 2}

num_outputs!{RoIAlignRotated, 1}

inputs!{RoIAlignRotated, 
    0 => ("X", "4D feature map input of shape (N, C, H, W)."),
    1 => ("RoIs", 
        "2D input of shape (R, 5 or 6) specifying R RoIs representing: batch index in [0, N - 1], 
        center_x, center_y, width, height, angle. The RoI coordinates are in the coordinate system of 
        the input image. `angle` should be specified in degrees and represents the RoI rotated counter-clockwise. 
        For inputs corresponding to a single image, batch index can be excluded to have just 5 columns.")
}

outputs!{RoIAlignRotated, 
    0 => ("Y", "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element is a pooled feature map cooresponding to the r-th RoI.")
}

args!{RoIAlignRotated, 
    0 => ("spatial_scale", "(float) default 1.0; Spatial scale of the input feature map X relative to the input image. E.g., 0.0625 if X has a stride of 16 w.r.t. the input image."),
    1 => ("pooled_h", "(int) default 1; Pooled output Y's height."),
    2 => ("pooled_w", "(int) default 1; Pooled output Y's width."),
    3 => ("sampling_ratio", "(int) default -1; number of sampling points in the interpolation grid used to compute the 
        output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. 
        If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / pooled_w), and likewise for height).")
}

impl<T,Context> RoIAlignRotatedOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            sampling_ratio_( this->template GetSingleArgument<int>("sampling_ratio", -1)),
            aligned_(this->template GetSingleArgument<bool>("aligned", false)) 

        DCHECK_GT(spatial_scale_, 0);
        DCHECK_GT(pooled_height_, 0);
        DCHECK_GT(pooled_width_, 0);
        DCHECK_GE(sampling_ratio_, 0);
        DCHECK(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}
