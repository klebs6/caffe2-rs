crate::ix!();

/**
  | Region of Interest (RoI) align operation
  | as used in Mask R-CNN.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIAlignOp<T, Context> {

    storage:         OperatorStorage,
    context:         Context,

    order:           StorageOrder,
    spatial_scale:   f32,
    pooled_h:        i32,
    pooled_w:        i32,
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

num_inputs!{RoIAlign, 2}

num_outputs!{RoIAlign, 1}

inputs!{RoIAlign, 
    0 => ("X",                "4D feature map input of shape (N, C, H, W)."),
    1 => ("RoIs",             "2D input of shape (R, 4 or 5) specifying R RoIs representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI coordinates are in the coordinate system of the input image. For inputs corresponding to a single image, batch index can be excluded to have just 4 columns.")
}

outputs!{RoIAlign, 
    0 => ("Y",                "4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element is a pooled feature map cooresponding to the r-th RoI.")
}

args!{RoIAlign, 
    0 => ("spatial_scale",   "(float) default 1.0; Spatial scale of the input feature map X relative to the input image. E.g., 0.0625 if X has a stride of 16 w.r.t. the input image."),
    1 => ("pooled_h",        "(int) default 1; Pooled output Y's height."),
    2 => ("pooled_w",        "(int) default 1; Pooled output Y's width."),
    3 => ("sampling_ratio",  "(int) default -1; number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / pooled_w), and likewise for height).")
}

impl<T, Context> RoIAlignOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NCHW"))),
            OP_SINGLE_ARG(float, "spatial_scale", spatial_scale_, 1.0f),
            OP_SINGLE_ARG(int, "pooled_h", pooled_h_, 1),
            OP_SINGLE_ARG(int, "pooled_w", pooled_w_, 1),
            OP_SINGLE_ARG(int, "sampling_ratio", sampling_ratio_, -1),
            OP_SINGLE_ARG(bool, "aligned", aligned_, false) 

        DCHECK_GT(spatial_scale_, 0.0f);
        DCHECK_GT(pooled_h_, 0);
        DCHECK_GT(pooled_w_, 0);
        DCHECK(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& R = Input(1);

        CAFFE_ENFORCE_EQ(X.dim(), 4);
        CAFFE_ENFORCE_EQ(R.dim(), 2);
        const int64_t roi_cols = R.size(1);
        CAFFE_ENFORCE(roi_cols == 4 || roi_cols == 5);
        const int64_t N = R.size(0);
        const int64_t C = X.size(order_ == StorageOrder::NCHW ? 1 : 3);
        const int64_t H = X.size(order_ == StorageOrder::NCHW ? 2 : 1);
        const int64_t W = X.size(order_ == StorageOrder::NCHW ? 3 : 2);
        const std::vector<int64_t> Y_sizes = order_ == StorageOrder::NCHW
            ? std::vector<int64_t>{N, C, pooled_h_, pooled_w_}
            : std::vector<int64_t>{N, pooled_h_, pooled_w_, C};

        auto* Y = Output(0, Y_sizes, at::dtype<T>());
        if (N == 0) {
          return true;
        }
        const T* X_data = X.template data<T>();
        const T* R_data = R.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        return order_ == StorageOrder::NCHW
            ? RunOnDeviceWithOrderNCHW(N, C, H, W, roi_cols, X_data, R_data, Y_data)
            : RunOnDeviceWithOrderNHWC(
                  N, C, H, W, roi_cols, X_data, R_data, Y_data);
        */
    }
}
