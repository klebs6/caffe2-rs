crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

/**
  | Applies gated linear unit to the input
  | Tensor X.
  | 
  | The output Y is half the size of the input
  | X, so if the shape of X is [d1, d2, ...,
  | N] shape of
  | 
  | Y will be [d1, d2, ..., dn/2] and Y(:dn-1,
  | i) = GLU(X(:dn-1, i), X(:dn-1, i+N/2))
  | = X(dn-1, i) sigmoid(X(dn-1, i+N/2))
  |
  */
pub struct GluOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    dim:     i32,
    phantom: PhantomData<T>,
}

num_inputs!{Glu, 1}

num_outputs!{Glu, 1}

inputs!{Glu, 
    0 => ("X", "1D input tensor")
}

outputs!{Glu, 
    0 => ("Y", "1D output tensor")
}

register_cpu_operator!{Glu, GluOp<f32, CPUContext>}

impl<T, Context> GluOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dim_(this->template GetSingleArgument<int>("dim", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        vector<int64_t> Yshape;
        Yshape.insert(Yshape.end(), X.sizes().begin(), X.sizes().end());
        const int split_index = dim_ == -1 ? Yshape.size() - 1 : dim_;
        CAFFE_ENFORCE(
            Yshape[split_index] % 2 == 0,
            "Split dimension ",
            Yshape[split_index],
            " should be divided by two");
        const int split_dim_size = Yshape[split_index] / 2;
        const int M = X.size_to_dim(split_index);
        const int N = X.size_from_dim(split_index + 1);
        Yshape[split_index] = split_dim_size;
        auto* Y = Output(0, Yshape, at::dtype<T>());
        ComputeGlu(
            M,
            split_dim_size,
            N,
            X.template data<T>(),
            Y->template mutable_data<T>());
        return true;
        */
    }
}

#[inline] pub fn sigmoid(x: f32) -> f32 {
    
    todo!();
    /*
        if (x >= 0) {
        return 1. / (1. + exp(-x));
      } else {
        const float exp_x = exp(x);
        return exp_x / (1 + exp_x);
      }
    */
}

impl GluOp<f32, CPUContext> {

    #[inline] pub fn compute_glu(
        &mut self, 
        m:         i32,
        split_dim: i32,
        n:         i32,
        xdata:     *const f32,
        ydata:     *mut f32)  
    {
        todo!();
        /*
            const int xStride = 2 * split_dim * N;
      const int yStride = split_dim * N;
      for (int i = 0; i < M; ++i) {
        const int idx = i * xStride;
        const int idy = i * yStride;
        for (int j = 0; j < split_dim; ++j) {
          const int jN = j * N;
          const int jdx1 = idx + jN;
          const int jdx2 = idx + (j + split_dim) * N;
          const int jdy = idy + jN;
          for (int k = 0; k < N; ++k) {
            const float x1 = Xdata[jdx1 + k];
            const float x2 = Xdata[jdx2 + k];
            Ydata[jdy + k] = x1 * sigmoid(x2);
          }
        }
      }
        */
    }
}
