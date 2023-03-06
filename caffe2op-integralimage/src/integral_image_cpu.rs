crate::ix!();

impl<T, Context> IntegralImageOp<T, Context> {
    
    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      CAFFE_ENFORCE_EQ(X.dim(), 4, "Only supports 4D tensors for the momement");

      vector<int64_t> out_shape(X.sizes().vec());
      out_shape[2] += 1; // H + 1 output size
      out_shape[3] += 1; // W + 1 output size
      auto* Y = Output(0, out_shape, at::dtype<float>());
      const int ind = X.dim32(0);
      const int chans = X.dim32(1);
      const int rows_in = X.dim32(2);
      const int cols_in = X.dim32(3);
      const int rows_out = Y->dim32(2);
      const int cols_out = Y->dim32(3);

      const float* input_data = X.template data<float>();
      float* output_data = Y->template mutable_data<float>();

      const int row_out_pass_size = ind * chans * rows_out;
      const int row_in_pass_size = ind * chans * rows_in;
      EigenMatrixMapRowMajor<float> Y_arr(output_data, row_out_pass_size, cols_out);
      ConstEigenMatrixMapRowMajor<float> X_arr(
          input_data, row_in_pass_size, cols_in);

      // Row Pass
      for (int i = 0; i < row_out_pass_size; i++) {
        int row = i % rows_out;
        int diff = i / rows_out + 1;
        Y_arr(i, 0) = 0.;
        if (row == 0) {
          for (int j = 1; j < cols_out; ++j) {
            Y_arr(i, j) = 0.;
          }
        } else {
          for (int j = 1; j < cols_out; ++j) {
            Y_arr(i, j) = Y_arr(i, j - 1) + X_arr(i - diff, j - 1);
          }
        }
      }

      // Col Pass
      const int col_out_pass_size = X.dim32(0) * chans * cols_out;
      for (int i = 0; i < col_out_pass_size; i++) {
        int col = i % cols_out;
        int row = i / cols_out;
        for (int j = row * rows_out + 1; j < (row + 1) * rows_out; ++j) {
          Y_arr(j, col) += Y_arr(j - 1, col);
        }
      }
      return true;
        */
    }
}
