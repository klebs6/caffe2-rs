crate::ix!();

impl<T,Context> IntegralImageGradientOp<T,Context> {
    
    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Original input to "forward" op
      auto& dY = Input(1); // Gradient of net w.r.t. output of "forward" op
      // (aka "gradOutput")
      auto* dX = Output(
          0, X.sizes(), at::dtype<float>()); // Gradient of net w.r.t. input to
                                             // "forward" op (aka "gradInput")

      const int ind = X.dim32(0);
      const int chans = X.dim32(1);
      const int rows_in = dY.dim32(2);
      const int cols_in = dY.dim32(3);
      const int rows_out = dX->dim32(2);
      const int cols_out = dX->dim32(3);

      const float* input_data = dY.template data<float>();
      float* output_data = dX->template mutable_data<float>();

      const int row_out_pass_size = ind * chans * rows_out;
      const int row_in_pass_size = ind * chans * rows_in;
      EigenMatrixMapRowMajor<float> dX_arr(
          output_data, row_out_pass_size, cols_out);
      ConstEigenMatrixMapRowMajor<float> dY_arr(
          input_data, row_in_pass_size, cols_in);
      Eigen::MatrixXf tmp(row_in_pass_size, cols_out);

      // Row Pass dY(N, C, H+1, W+1) => tmp(N, C, H+1, W)
      for (int i = 0; i < row_in_pass_size; i++) {
        tmp(i, 0) = dY_arr(i, 0);
        for (int j = 1; j < cols_out; ++j) {
          tmp(i, j) = tmp(i, j - 1) + dY_arr(i, j);
        }
      }

      // Col Pass tmp(N, C, H+1, W)=>dX(N, C, H, W)
      const int col_out_pass_size = X.dim32(0) * chans * cols_out;
      for (int i = 0; i < col_out_pass_size; i++) {
        int col = i % cols_out;
        int row_out_start = (i / cols_out) * rows_out;
        int row_in_start = (i / cols_out) * rows_in;
        dX_arr(row_out_start, col) = tmp(row_in_start, col);
        for (int j = 1; j < rows_out; ++j) {
          dX_arr(row_out_start + j, col) =
              dX_arr(row_out_start + j - 1, col) + tmp(row_in_start + j, col);
        }
      }
      return true;
        */
    }
}
