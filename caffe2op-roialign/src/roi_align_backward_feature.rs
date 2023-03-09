crate::ix!();

#[inline] pub fn roi_align_backward_feature<T>(
    nthreads:               i32,
    top_diff:               *const T,
    num_rois:               i32,
    spatial_scale:          &T,
    channels:               i32,
    height:                 i32,
    width:                  i32,
    pooled_height:          i32,
    pooled_width:           i32,
    sampling_ratio:         i32,
    bottom_diff:            *mut T,
    bottom_rois:            *const T,
    rois_cols:              i32,
    continuous_coordinate:  bool) 
{
    todo!();
    /*
        DCHECK(rois_cols == 4 || rois_cols == 5);

      for (int index = 0; index < nthreads; index++) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_bottom_rois = bottom_rois + n * rois_cols;
        int roi_batch_ind = 0;
        if (rois_cols == 5) {
          roi_batch_ind = offset_bottom_rois[0];
          offset_bottom_rois++;
        }

        // Do not using rounding; this implementation detail is critical
        T roi_offset = continuous_coordinate ? T(0.5) : 0;
        T roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
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

        T* offset_bottom_diff =
            bottom_diff + (roi_batch_ind * channels + c) * height * width;

        int top_offset = (n * channels + c) * pooled_height * pooled_width;
        const T* offset_top_diff = top_diff + top_offset;
        const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          const T y = roi_start_h + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const T x = roi_start_w + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);

            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            bilinear_interpolate_gradient(
                height,
                width,
                y,
                x,
                w1,
                w2,
                w3,
                w4,
                x_low,
                x_high,
                y_low,
                y_high,
                index);

            T g1 = top_diff_this_bin * w1 / count;
            T g2 = top_diff_this_bin * w2 / count;
            T g3 = top_diff_this_bin * w3 / count;
            T g4 = top_diff_this_bin * w4 / count;

            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
              // atomic add is not needed for now since it is single threaded
              add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
              add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
              add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
              add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
            } // if
          } // ix
        } // iy
      } // for
    */
}
