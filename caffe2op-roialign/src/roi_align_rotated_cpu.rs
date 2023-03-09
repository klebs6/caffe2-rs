crate::ix!();

register_cpu_operator!{RoIAlignRotated, RoIAlignRotatedOp<float, CPUContext>}

pub type RoIAlignRotatedOpFloatCPU = RoIAlignRotatedOp<f32,CPUContext>;

export_caffe2_op_to_c10_cpu!{
    RoIAlignRotated,
    "_caffe2::RoIAlignRotated(
        Tensor features, 
        Tensor rois, 
        str order, 
        float spatial_scale, 
        int pooled_h, 
        int pooled_w, 
        int sampling_ratio, 
        bool aligned) -> Tensor",
    RoIAlignRotatedOpFloatCPU
}

impl RoIAlignRotatedOp<f32, CPUContext> {

    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Input data to pool
      auto& R = Input(1); // RoIs

      if (R.numel() == 0) {
        std::vector<int64_t> sizes;
        // Handle empty rois
        if (order_ == StorageOrder::NCHW) {
          sizes = {0, X.dim32(1), pooled_height_, pooled_width_};
        } else if (order_ == StorageOrder::NHWC) {
          sizes = {0, pooled_height_, pooled_width_, X.dim32(3)};
        }
        // Output tensor is inititalized with proper sizes and data type
        Output(0, sizes, at::dtype<float>());
        return true;
      }

      CAFFE_ENFORCE_EQ(R.dim(), 2);
      // Each element of R is [batch_id center_x center_y width height angle].
      // If R has 6 columns, the first column is the index, otherwise 0.
      CAFFE_ENFORCE(R.dim32(1) == 5 || R.dim32(1) == 6);

      assert(sampling_ratio_ >= 0);

      if (order_ == StorageOrder::NCHW) {
        auto* Y = Output(
            0,
            {R.dim32(0), X.dim32(1), pooled_height_, pooled_width_},
            at::dtype<float>()); // RoI pooled data

        size_t output_size = Y->numel();
        ROIAlignRotatedForward<float>(
            output_size,
            X.data<float>(),
            spatial_scale_,
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            R.data<float>(),
            R.dim32(1),
            Y->mutable_data<float>(),
            order_,
            aligned_);
      } else if (order_ == StorageOrder::NHWC) {
        auto* Y = Output(
            0,
            {R.dim32(0), pooled_height_, pooled_width_, X.dim32(3)},
            at::dtype<float>()); // RoI pooled data
        size_t output_size = Y->numel();
        ROIAlignRotatedForward<float>(
            output_size,
            X.data<float>(),
            spatial_scale_,
            X.dim32(3),
            X.dim32(1),
            X.dim32(2),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            R.data<float>(),
            R.dim32(1),
            Y->mutable_data<float>(),
            order_,
            aligned_);
      }

      return true;
        */
    }
}
