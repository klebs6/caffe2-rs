crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Pooling.cpp]

pub fn check1d(
        function_name: *const u8,
        argument_name: *const u8,
        x:             &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(
          x.size() == 1,
          function_name, "() argument '", argument_name,
          "' should contain one int (got ", x.size(), ")");
        */
}

pub fn adaptive_avg_pool1d(
        self_:       &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            checkDim("adaptive_avg_pool1d", TensorArg(self, "self", 1), 3);
      check1d("adaptive_avg_pool1d", "output_size", output_size);

      auto output = adaptive_avg_pool2d(
          self.unsqueeze(2),
          {1, output_size[0]});

      return output.squeeze(2);
        */
}

pub fn adaptive_max_pool1d(
        self_:       &Tensor,
        output_size: &[i32]) -> (Tensor,Tensor) {
    
    todo!();
        /*
            checkDim("adaptive_max_pool1d", TensorArg(self, "self", 1), 3);
      check1d("adaptive_max_pool1d", "output_size", output_size);

      Tensor output, indices;
      tie(output, indices) = adaptive_max_pool2d(
          self.unsqueeze(2),
          {1, output_size[0]});

      return make_tuple(output.squeeze(2), indices.squeeze(2));
        */
}


pub fn max_pool1d_with_indices(
        self_:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            if (stride.empty()) {
        stride = kernel_size;
      }
      checkDim("max_pool1d", TensorArg(self, "self", 1), 3);
      check1d("max_pool1d", "kernel_size", kernel_size);
      check1d("max_pool1d", "stride", stride);
      check1d("max_pool1d", "padding", padding);
      check1d("max_pool1d", "dilation", dilation);

      NoNamesGuard guard;

      Tensor output, indices;
      tie(output, indices) = max_pool2d_with_indices(
          self.unsqueeze(2),
          {1, kernel_size[0]},
          {1, stride[0]},
          {0, padding[0]},
          {1, dilation[0]},
          ceil_mode);

      output  = output.squeeze(2);
      indices = indices.squeeze(2);

      guard.reset();
      namedinference::propagate_names(output, self);
      namedinference::propagate_names(indices, self);

      return make_tuple(output, indices);
        */
}



pub fn avg_pool1d(
        self_:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool) -> Tensor {
    
    todo!();
        /*
            if (stride.empty()) {
        stride = kernel_size;
      }
      checkDim("avg_pool1d", TensorArg(self, "self", 1), 3);
      check1d("avg_pool1d", "kernel_size", kernel_size);
      check1d("avg_pool1d", "stride", stride);
      check1d("avg_pool1d", "padding", padding);

      auto output = avg_pool2d(
          self.unsqueeze(2),
          {1, kernel_size[0]},
          {1, stride[0]},
          {0, padding[0]},
          ceil_mode,
          count_include_pad);

      return output.squeeze(2);
        */
}


pub fn max_pool2d(
        self_:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            if (self.is_quantized()) {
        return quantized_max_pool2d(self, kernel_size, stride, padding,
                                        dilation, ceil_mode);
      }
      if (self.is_mkldnn()) {
        return mkldnn_max_pool2d(
            self, kernel_size, stride, padding, dilation, ceil_mode);
      }

    #if defined(C10_MOBILE)
      if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                                 dilation, ceil_mode)) {
        return xnnpack::max_pool2d(
            self, kernel_size, padding, stride, dilation, ceil_mode);
      }
    #endif
      auto output_and_indices = max_pool2d_with_indices(
          self, kernel_size, stride, padding, dilation, ceil_mode);
      return get<0>(output_and_indices);
        */
}


pub fn max_pool3d(
        self_:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            if (self.is_mkldnn()) {
        return mkldnn_max_pool3d(
            self, kernel_size, stride, padding, dilation, ceil_mode);
      }
      auto output_and_indices = max_pool3d_with_indices(
          self, kernel_size, stride, padding, dilation, ceil_mode);
      return get<0>(output_and_indices);
        */
}
