crate::ix!();

impl RoIAlignGradientOp<f32, CPUContext> {

    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Input data to pool
      auto& R = Input(1); // RoIs
      auto& dY = Input(2); // Gradient of net w.r.t. output of "forward" op
                           // (aka "gradOutput")

      CAFFE_ENFORCE_EQ(R.dim(), 2);
      // if R has 5 columns, the first column is the index, otherwise 0
      CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

      auto* dX = Output(
          0,
          X.sizes(),
          at::dtype<float>()); // Gradient of net w.r.t. input to "forward" op (aka
                               // "gradInput")

      // Must zero-out dX before accumulating gradients
      // (TODO): Kaiming - is this safe?
      math::Set<float, CPUContext>(
          dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);

      if (dY.numel() > 0) { // Handle possibly empty gradient if there were no rois
        ROIAlignBackwardFeature<float>(
            dY.numel(),
            dY.data<float>(),
            R.dim32(0),
            spatial_scale_,
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            dX->template mutable_data<float>(),
            R.data<float>(),
            R.dim32(1),
            aligned_);
      }
      return true;
        */
    }
}

register_cpu_operator!{
    RoIAlignGradient, 
    RoIAlignGradientOp<float, CPUContext>
}
