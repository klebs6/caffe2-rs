crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
    CPUContext,
};

/**
  | This op computes the $L_p$ norm of the
  | one dimensional input tensor $X$, and
  | outputs a one dimensional output tensor
  | $Y$. Here, the $L_p$ norm is calculated
  | as
  | 
  | $$L_p(\mathbf{x}) = \sum_i x_i^p$$
  | 
  | This op supports $p$ values of 1 or 2.
  | If the average argument is set, the norm
  | is calculated as
  | 
  | Lp_averaged_norm(x) is defined as
  | 
  | Lp_averaged_norm(x) = LpNorm(x) /
  | size(x).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.h
  | //- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc
  |
  */
pub struct LpNormOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    p:       i32,
    average: bool,
    phantom: PhantomData<T>,
}

num_inputs!{LpNorm, 1}

num_outputs!{LpNorm, 1}

inputs!{LpNorm, 
    0 => ("X", "1D Input tensor of data to be operated on.")
}

outputs!{LpNorm, 
    0 => ("Z", "1D output tensor")
}

args!{LpNorm, 
    0 => ("p",       "*(type: int; default: 2, possible values: {1,2})* Order of the norm in p-norm."),
    1 => ("average", "*(type: bool; default: False)* Whether we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x)")
}

tensor_inference_function!{LpNorm, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<int64_t> output_dims(1);
      output_dims[0] = 1; // 1
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    } */
}

impl<T, Context> LpNormOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "p", p_, 2),
            OP_SINGLE_ARG(bool, "average", average_, false) 

        CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
        */
    }
}

impl LpNormOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      auto* norm = Output(0, {1}, at::dtype<float>());
      const float* X_data = X.data<float>();
      const float size = average_ ? (float)X.numel() : 1.0f;
      CAFFE_ENFORCE_GT(size, 0);
      if (p_ == 1) {
        *(norm->template mutable_data<float>()) =
            (ConstEigenVectorMap<float>(X_data, X.numel()).array()).abs().sum() /
            size;
        // L1(x) = sum(|x|), L1_average(x) = sum(\x\) / x.size()
      } else if (p_ == 2) {
        *(norm->template mutable_data<float>()) =
            (ConstEigenVectorMap<float>(X_data, X.numel()).array()).square().sum() /
            size;
        // L2(x) = (sum(|x|^2)), L2_average(x) = sum(|x|^2) / x.size()
      }
      return true;
        */
    }
}

/**
  | Given one input float tensor X, derivative
  | dout, and produces one output float
  | tensor dX.
  | 
  | dX is the derivative of the Lp norm of
  | tensor X, computed as dx = d(sum over
  | |x^p|)/dx, in which p is either 1 or 2(currently
  | only supports l1 and l2 norm) determined
  | by the argument p.
  |
  */
pub struct LpNormGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    p:       i32,
    average: bool,
    phantom: PhantomData<T>,
}

num_inputs!{LpNormGradient, 2}

num_outputs!{LpNormGradient, 1}

inputs!{LpNormGradient, 
    0 => ("X", "1D input tensor"),
    1 => ("dout", "1D input tensor")
}

outputs!{LpNormGradient, 
    0 => ("dx", "1D output tensor")
}

args!{LpNormGradient, 
    0 => ("p", "Order of the norm in p-norm"),
    1 => ("average", "whehther we calculate norm or averaged_norm. The Lp_averaged_norm(x) is defined as Lp_averaged_normgradient(x) = LpNormGradient(x) / size(x)")
}

impl<T, Context> LpNormGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "p", p_, 2),
            OP_SINGLE_ARG(bool, "average", average_, false) 

        CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
        */
    }
}

impl LpNormGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& dnorm = Input(1);

      CAFFE_ENFORCE_EQ(dnorm.dim(), 1);
      CAFFE_ENFORCE_EQ(dnorm.dim32(0), 1);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      const float size = average_ ? (float)X.numel() : 1.0f;
      if (p_ == 1) {
        EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
            .array() = ConstEigenVectorMap<float>(X.data<float>(), X.numel())
                           .array()
                           .unaryExpr([](float x) {
                             const float kEps = 1e-12f;
                             if (x < -kEps) {
                               return -1.0f;
                             } else if (x > kEps) {
                               return 1.0f;
                             } else {
                               return 0.0f;
                             }
                           }) *
            ((dnorm.data<float>())[0] / size);
      } else if (p_ == 2) {
        EigenVectorMap<float>(dX->template mutable_data<float>(), X.numel())
            .array() =
            ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array() * 2.0f *
            ((dnorm.data<float>())[0] / size);
      }

      return true;
        */
    }
}

// LpNorm
register_cpu_operator!{LpNorm,         LpNormOp<f32, CPUContext>}
register_cpu_operator!{LpNormGradient, LpNormGradientOp<f32, CPUContext>}

#[test] fn lp_norm_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LpNorm",
        ["X"],
        ["Y"],
        p=2
    )
    X = np.array([5., 2.])
    print("X:\n",X)

    // Feed X into workspace
    workspace.FeedBlob("X", X.astype(np.float32))

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [5. 2.]
    Y:
     [29.]

    */
}

pub struct GetLpNormGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLpNormGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LpNormGradient",
            "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LpNorm, GetLpNormGradient}
