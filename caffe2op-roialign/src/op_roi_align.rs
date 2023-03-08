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

pub struct BilinearInterpolationParam<T> {
    p1: i64,
    p2: i64,
    p3: i64,
    p4: i64,
    w1: T,
    w2: T,
    w3: T,
    w4: T,
}

#[inline] pub fn make_bilinear_interpolation_params<T>(
    h:             i64,
    w:             i64,
    pooled_h:      i64,
    pooled_w:      i64,
    bin_size_h:    T,
    bin_size_w:    T,
    bin_grid_h:    i64,
    bin_grid_w:    i64,
    roi_start_h:   T,
    roi_start_w:   T) -> Vec<BilinearInterpolationParam<T>> 
{
    todo!();
    /*
        std::vector<BilinearInterpolationParam<T>> params(
          pooled_h * pooled_w * bin_grid_h * bin_grid_w);
      const T ch = bin_size_h / static_cast<T>(bin_grid_h);
      const T cw = bin_size_w / static_cast<T>(bin_grid_w);
      int64_t cnt = 0;
      for (int64_t ph = 0; ph < pooled_h; ++ph) {
        for (int64_t pw = 0; pw < pooled_w; ++pw) {
          for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
            const T yy = roi_start_h + static_cast<T>(ph) * bin_size_h +
                (static_cast<T>(iy) + T(0.5)) * ch;
            if (yy < T(-1) || yy > static_cast<T>(H)) {
              std::memset(params.data() + cnt, 0, bin_grid_w * sizeof(params[0]));
              cnt += bin_grid_w;
              continue;
            }
            for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
              const T xx = roi_start_w + pw * bin_size_w +
                  (static_cast<T>(ix) + T(0.5f)) * cw;
              BilinearInterpolationParam<T>& param = params[cnt++];
              if (xx < T(-1) || xx > static_cast<T>(W)) {
                std::memset(&param, 0, sizeof(param));
                continue;
              }
              const T y = std::min(std::max(yy, T(0)), static_cast<T>(H - 1));
              const T x = std::min(std::max(xx, T(0)), static_cast<T>(W - 1));
              const int64_t yl = static_cast<int64_t>(std::floor(y));
              const int64_t xl = static_cast<int64_t>(std::floor(x));
              const int64_t yh = std::min(yl + 1, H - 1);
              const int64_t xh = std::min(xl + 1, W - 1);
              const T py = y - static_cast<T>(yl);
              const T px = x - static_cast<T>(xl);
              const T qy = T(1) - py;
              const T qx = T(1) - px;
              param.p1 = yl * W + xl;
              param.p2 = yl * W + xh;
              param.p3 = yh * W + xl;
              param.p4 = yh * W + xh;
              param.w1 = qy * qx;
              param.w2 = qy * px;
              param.w3 = py * qx;
              param.w4 = py * px;
            }
          }
        }
      }
      return params;
    */
}

impl RoIAlignOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_order_nchw(
        &mut self, 
        n:          i64,
        c:          i64,
        h:          i64,
        w:          i64,
        roi_cols:   i64,
        x:          *const f32,
        r:          *const f32,
        y:          *mut f32) -> bool 
    {
        todo!();
        /*
            DCHECK(roi_cols == 4 || roi_cols == 5);
          const float roi_offset = aligned_ ? 0.5f : 0.0f;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
          for (int64_t n = 0; n < N; ++n) {
            const int64_t roi_batch_idx = roi_cols == 4 ? 0 : R[n * roi_cols];
            const float* X_ptr = X + roi_batch_idx * C * H * W;
            const float* R_ptr = R + n * roi_cols + (roi_cols == 5);
            float* Y_ptr = Y + n * C * pooled_h_ * pooled_w_;

            // Do not using rounding; this implementation detail is critical
            const float roi_w1 = R_ptr[0] * spatial_scale_ - roi_offset;
            const float roi_h1 = R_ptr[1] * spatial_scale_ - roi_offset;
            const float roi_w2 = R_ptr[2] * spatial_scale_ - roi_offset;
            const float roi_h2 = R_ptr[3] * spatial_scale_ - roi_offset;
            float roi_w = roi_w2 - roi_w1;
            float roi_h = roi_h2 - roi_h1;
            if (aligned_) {
              CAFFE_ENFORCE(
                  roi_w >= 0.0f && roi_h >= 0.0f,
                  "ROIs in ROIAlign do not have non-negative size!");
            } else { // backward compatibility
              // Force malformed ROIs to be 1x1
              roi_w = std::max(roi_w, 1.0f);
              roi_h = std::max(roi_h, 1.0f);
            }
            const float bin_size_h = roi_h / static_cast<float>(pooled_h_);
            const float bin_size_w = roi_w / static_cast<float>(pooled_w_);

            // We use roi_bin_grid to sample the grid and mimic integral
            const int64_t bin_grid_h = (sampling_ratio_ > 0)
                ? sampling_ratio_
                : static_cast<int64_t>(ceil(roi_h / static_cast<float>(pooled_h_)));
            const int64_t bin_grid_w = (sampling_ratio_ > 0)
                ? sampling_ratio_
                : static_cast<int64_t>(ceil(roi_w / static_cast<float>(pooled_w_)));

            const std::vector<BilinearInterpolationParam<float>> params =
                MakeBilinearInterpolationParams(
                    H,
                    W,
                    pooled_h_,
                    pooled_w_,
                    bin_size_h,
                    bin_size_w,
                    bin_grid_h,
                    bin_grid_w,
                    roi_h1,
                    roi_w1);

            const float scale = 1.0f / static_cast<float>(bin_grid_h * bin_grid_w);
            for (int64_t c = 0; c < C; ++c) {
              int64_t cnt = 0;
              for (int64_t ph = 0; ph < pooled_h_; ++ph) {
                for (int64_t pw = 0; pw < pooled_w_; ++pw) {
                  float sum = 0.0f;
                  for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
                    for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
                      const BilinearInterpolationParam<float>& param = params[cnt++];
                      sum += param.w1 * X_ptr[param.p1] + param.w2 * X_ptr[param.p2] +
                          param.w3 * X_ptr[param.p3] + param.w4 * X_ptr[param.p4];
                    }
                  }
                  Y_ptr[ph * pooled_w_ + pw] = sum * scale;
                }
              }
              X_ptr += H * W;
              Y_ptr += pooled_h_ * pooled_w_;
            }
          }

          return true;
        */
    }

    #[inline] pub fn run_on_device_with_order_nhwc(
        &mut self,
        n:         i64,
        c:         i64,
        h:         i64,
        w:         i64,
        roi_cols:  i64,
        x:         *const f32,
        r:         *const f32,
        y:         *mut f32) -> bool 
    {

        todo!();
        /*
            DCHECK(roi_cols == 4 || roi_cols == 5);
          const float roi_offset = aligned_ ? 0.5f : 0.0f;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
          for (int64_t n = 0; n < N; ++n) {
            const int64_t roi_batch_idx = roi_cols == 4 ? 0 : R[n * roi_cols];
            const float* X_ptr = X + roi_batch_idx * C * H * W;
            const float* R_ptr = R + n * roi_cols + (roi_cols == 5);
            float* Y_ptr = Y + n * C * pooled_h_ * pooled_w_;

            // Do not using rounding; this implementation detail is critical
            const float roi_w1 = R_ptr[0] * spatial_scale_ - roi_offset;
            const float roi_h1 = R_ptr[1] * spatial_scale_ - roi_offset;
            const float roi_w2 = R_ptr[2] * spatial_scale_ - roi_offset;
            const float roi_h2 = R_ptr[3] * spatial_scale_ - roi_offset;
            float roi_w = roi_w2 - roi_w1;
            float roi_h = roi_h2 - roi_h1;
            if (aligned_) {
              CAFFE_ENFORCE(
                  roi_w >= 0.0f && roi_h >= 0.0f,
                  "ROIs in ROIAlign do not have non-negative size!");
            } else { // backward compatibility
              // Force malformed ROIs to be 1x1
              roi_w = std::max(roi_w, 1.0f);
              roi_h = std::max(roi_h, 1.0f);
            }
            const float bin_size_h = roi_h / static_cast<float>(pooled_h_);
            const float bin_size_w = roi_w / static_cast<float>(pooled_w_);

            // We use roi_bin_grid to sample the grid and mimic integral
            const int64_t bin_grid_h = (sampling_ratio_ > 0)
                ? sampling_ratio_
                : static_cast<int64_t>(ceil(roi_h / static_cast<float>(pooled_h_)));
            const int64_t bin_grid_w = (sampling_ratio_ > 0)
                ? sampling_ratio_
                : static_cast<int64_t>(ceil(roi_w / static_cast<float>(pooled_w_)));

            const std::vector<BilinearInterpolationParam<float>> params =
                MakeBilinearInterpolationParams(
                    H,
                    W,
                    pooled_h_,
                    pooled_w_,
                    bin_size_h,
                    bin_size_w,
                    bin_grid_h,
                    bin_grid_w,
                    roi_h1,
                    roi_w1);

            const float scale = 1.0f / static_cast<float>(bin_grid_h * bin_grid_w);
            int64_t cnt = 0;
            for (int64_t ph = 0; ph < pooled_h_; ++ph) {
              for (int64_t pw = 0; pw < pooled_w_; ++pw) {
                EigenVectorArrayMap<float> Y_arr(Y_ptr + (ph * pooled_w_ + pw) * C, C);
                Y_arr.setZero();
                for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
                  for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
                    const BilinearInterpolationParam<float>& param = params[cnt++];
                    ConstEigenVectorArrayMap<float> x1_arr(X_ptr + param.p1 * C, C);
                    ConstEigenVectorArrayMap<float> x2_arr(X_ptr + param.p2 * C, C);
                    ConstEigenVectorArrayMap<float> x3_arr(X_ptr + param.p3 * C, C);
                    ConstEigenVectorArrayMap<float> x4_arr(X_ptr + param.p4 * C, C);
                    Y_arr += param.w1 * x1_arr + param.w2 * x2_arr + param.w3 * x3_arr +
                        param.w4 * x4_arr;
                  }
                }
                Y_arr *= scale;
              }
            }
          }

          return true;
        */
    }
}

register_cpu_operator!{RoIAlign, RoIAlignOp<f32, CPUContext>}

pub type RoIAlignCPUOp<T> = RoIAlignOp<T, CPUContext>;
