crate::ix!();

#[inline] pub fn roi_align_rotated_forward<T>(
    nthreads:              i32,
    bottom_data:           *const T,
    spatial_scale:         &T,
    channels:              i32,
    height:                i32,
    width:                 i32,
    pooled_height:         i32,
    pooled_width:          i32,
    sampling_ratio:        i32,
    bottom_rois:           *const T,
    roi_cols:              i32,
    top_data:              *mut T,
    order:                 StorageOrder,
    continuous_coordinate: bool)  {

    todo!();
    /*
        DCHECK(roi_cols == 5 || roi_cols == 6);

      int n_rois = nthreads / channels / pooled_width / pooled_height;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int n = 0; n < n_rois; n++) {
        int index_n = n * channels * pooled_width * pooled_height;
        // roi could have 5 or 6 columns
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = 0;
        if (roi_cols == 6) {
          roi_batch_ind = offset_bottom_rois[0];
          offset_bottom_rois++;
        }

        // Do not round
        T roi_offset = continuous_coordinate ? T(0.5) : 0;
        T roi_center_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
        T roi_center_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
        T roi_width = offset_bottom_rois[2] * spatial_scale;
        T roi_height = offset_bottom_rois[3] * spatial_scale;
        T theta = offset_bottom_rois[4] * M_PI / 180.0;

        if (continuous_coordinate) {
          CAFFE_ENFORCE(
              roi_width >= 0 && roi_height >= 0,
              "ROIs in ROIAlign do not have non-negative size!");
        } else { // backward compatibility
          // Force malformed ROIs to be 1x1
          roi_width = std::max(roi_width, (T)1.);
          roi_height = std::max(roi_height, (T)1.);
        }
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        // We want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization.
        std::vector<PreCalc<T>> pre_calc(
            roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);

        // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
        // Appropriate translation needs to be applied after.
        T roi_start_h = -roi_height / 2.0;
        T roi_start_w = -roi_width / 2.0;
        pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_center_h,
            roi_center_w,
            theta,
            pre_calc);

        if (order == StorageOrder::NCHW) {
          for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * pooled_width * pooled_height;
            const T* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
              for (int pw = 0; pw < pooled_width; pw++) {
                int index = index_n_c + ph * pooled_width + pw;

                T output_val = 0.;
                for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                  for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    PreCalc<T> pc = pre_calc[pre_calc_index];
                    output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                        pc.w2 * offset_bottom_data[pc.pos2] +
                        pc.w3 * offset_bottom_data[pc.pos3] +
                        pc.w4 * offset_bottom_data[pc.pos4];

                    pre_calc_index += 1;
                  }
                }
                output_val /= count;

                top_data[index] = output_val;
              } // for pw
            } // for ph
          } // for c
        } // if nchw

        if (order == StorageOrder::NHWC) {
          const T* offset_bottom_data =
              bottom_data + roi_batch_ind * channels * height * width;
          int pre_calc_index = 0;

          for (int ph = 0; ph < pooled_height; ph++) {
            for (int pw = 0; pw < pooled_width; pw++) {
              EVecXf output_vals = EVecXf::Zero(channels);

              for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                  PreCalc<T> pc = pre_calc[pre_calc_index];

                  ConstEigenVectorMap<T> data_1(
                      offset_bottom_data + channels * pc.pos1, channels);
                  ConstEigenVectorMap<T> data_2(
                      offset_bottom_data + channels * pc.pos2, channels);
                  ConstEigenVectorMap<T> data_3(
                      offset_bottom_data + channels * pc.pos3, channels);
                  ConstEigenVectorMap<T> data_4(
                      offset_bottom_data + channels * pc.pos4, channels);

                  output_vals += pc.w1 * data_1 + pc.w2 * data_2 + pc.w3 * data_3 +
                      pc.w4 * data_4;

                  pre_calc_index += 1;
                }
              }
              output_vals /= count;

              int index_nhw = index_n + (ph * pooled_width + pw) * channels;
              std::memcpy(
                  top_data + index_nhw, output_vals.data(), channels * sizeof(T));
            } // for pw
          } // for ph
        } // if nhwc
      } // for n
    */
}
