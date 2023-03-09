crate::ix!();

impl RoIPoolOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0); // Input data to pool
      const auto& R = Input(1); // RoIs
      auto* Y = Output(0); // RoI pooled data
      auto* A = is_test_ ? nullptr : Output(1); // argmaxes

      // Each ROI is of the form [batch_index x1 y1 x2 y2]
      CAFFE_ENFORCE_EQ(R.dim32(1), 5);

      // TODO: Handle the storage_order properly to get the NCWH.
      int batch_size = X.dim32(0);
      int channels = X.dim32(1);
      int height = X.dim32(2);
      int width = X.dim32(3);
      int num_rois = R.dim32(0);

      Y->Resize(num_rois, channels, pooled_height_, pooled_width_);
      if (!is_test_) {
        A->Resize(Y->sizes());
      }

      const float* Xdata = X.data<float>();
      const float* rois = R.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      int* argmax_data = is_test_ ? nullptr : A->template mutable_data<int>();

      // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
      for (int n = 0; n < num_rois; ++n) {
        int roi_batch_id = rois[0];
        int roi_start_w = round(rois[1] * spatial_scale_);
        int roi_start_h = round(rois[2] * spatial_scale_);
        int roi_end_w = round(rois[3] * spatial_scale_);
        int roi_end_h = round(rois[4] * spatial_scale_);
        CAFFE_ENFORCE_GE(roi_batch_id, 0);
        CAFFE_ENFORCE_LT(roi_batch_id, batch_size);

        // Force malformed ROIs to be 1x1
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);

        const float bin_size_h =
            static_cast<float>(roi_height) / static_cast<float>(pooled_height_);
        const float bin_size_w =
            static_cast<float>(roi_width) / static_cast<float>(pooled_width_);

        const float* batch_data = Xdata + roi_batch_id * X.size_from_dim(1);

        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Compute pooling region for this output unit:
              //  start (included) = floor(ph * roi_height / pooled_height_)
              //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
              int hstart =
                  static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
              int wstart =
                  static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
              int hend =
                  static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
              int wend =
                  static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

              // Add roi offsets and clip to input boundaries
              hstart = min(max(hstart + roi_start_h, 0), height);
              hend = min(max(hend + roi_start_h, 0), height);
              wstart = min(max(wstart + roi_start_w, 0), width);
              wend = min(max(wend + roi_start_w, 0), width);

              const int pool_index = ph * pooled_width_ + pw;

              // Define an empty pooling region to be zero
              bool is_empty = (hend <= hstart) || (wend <= wstart);
              Ydata[pool_index] = is_empty ? 0 : -FLT_MAX;
              if (!is_test_) {
                // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                argmax_data[pool_index] = -1;
              }

              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width + w;
                  if (batch_data[index] > Ydata[pool_index]) {
                    Ydata[pool_index] = batch_data[index];
                    if (!is_test_) {
                      argmax_data[pool_index] = index;
                    }
                  }
                }
              }
            }
          }
          // Increment all data pointers by one channel
          batch_data += X.size_from_dim(2);
          Ydata += Y->size_from_dim(2);
          if (!is_test_) {
            argmax_data += A->size_from_dim(2);
          }
        }
        // Increment ROI data pointer
        rois += R.size_from_dim(1);
      }

      return true;
        */
    }
}
