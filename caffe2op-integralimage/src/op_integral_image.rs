/*
   | template <typename T> 
   | using EigenMatrixMapRowMajor 
   | = Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
   |
   | template <typename T> 
   | using ConstEigenMatrixMapRowMajor 
   | = Eigen::Map< const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
   */

crate::ix!();

/**
  | Computes an integral image, which contains
  | the sum of pixel values within an image
  | vertically and horizontally.
  | 
  | This integral image can then be used
  | with other detection and tracking techniques.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct IntegralImageOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T,Context> IntegralImageOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

pub struct IntegralImageGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage:         OperatorStorage,
    context:         Context,
    row_pass_buffer: Tensor,

    // Input: X, dY (aka "gradOutput"); 
    // Output: dX (aka "gradInput")

    phantom: PhantomData<T>,
}

impl<T, Context> IntegralImageGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

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

register_cpu_operator!{IntegralImage,         IntegralImageOp<f32, CPUContext>}

register_cpu_operator!{IntegralImageGradient, IntegralImageGradientOp<f32, CPUContext>}

// Input: X; Output: Y
num_inputs!{IntegralImage, 1}

num_outputs!{IntegralImage, 1}

inputs!{IntegralImage, 
    0 => ("X", "Images tensor of the form (N, C, H, W)")
}

outputs!{IntegralImage, 
    0 => ("Y", "Integrated image of the form (N, C, H+1, W+1)")
}

num_inputs!{IntegralImageGradient, 2}

num_outputs!{IntegralImageGradient, 1}

pub struct GetIntegralImageGradient {}

impl GetGradientDefs for GetIntegralImageGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "IntegralImageGradient",
            "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    IntegralImage, 
    GetIntegralImageGradient
}
