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

register_cpu_operator!{IntegralImage,         IntegralImageOp<f32, CPUContext>}

// Input: X; Output: Y
num_inputs!{IntegralImage, 1}

num_outputs!{IntegralImage, 1}

inputs!{IntegralImage, 
    0 => ("X", "Images tensor of the form (N, C, H, W)")
}

outputs!{IntegralImage, 
    0 => ("Y", "Integrated image of the form (N, C, H+1, W+1)")
}

// template <typename T> 
// using EigenMatrixMapRowMajor 
// = Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// 
// template <typename T> 
// using ConstEigenMatrixMapRowMajor 
// = Eigen::Map< const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
