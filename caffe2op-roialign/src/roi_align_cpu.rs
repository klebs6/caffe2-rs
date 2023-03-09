crate::ix!();

register_cpu_operator!{RoIAlign, RoIAlignOp<f32, CPUContext>}

pub type RoIAlignCPUOp<T> = RoIAlignOp<T, CPUContext>;

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
