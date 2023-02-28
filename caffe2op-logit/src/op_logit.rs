crate::ix!();

use crate::{
    UnaryElementwiseWithArgsOp,
    CPUContext,
    GradientMakerBase,
    TensorTypes,
    OperatorStorage,
    OperatorDef,
};

/**
  | Elementwise logit transform: logit(x)
  | = log(x / (1 - x)), where x is the input 
  | data clampped in (eps, 1-eps).
  |
  */
pub struct LogitFunctor<Context> {
    eps: f32,

    phantom: PhantomData<Context>,

}

num_inputs!{Logit, 1}

num_outputs!{Logit, 1}

inputs!{Logit, 
    0 => ("X", "input float tensor")
}

outputs!{Logit, 
    0 => ("Y", "output float tensor")
}

args!{Logit, 
    0 => ("eps (optional)", "small positive epsilon value, the default is 1e-6.")
}

identical_type_and_shape!{Logit}

allow_inplace!{Logit, vec![(0, 0)]}

impl<Context> LogitFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : eps_(op.GetSingleArgument<float>("eps", 1e-6f)) 

        CAFFE_ENFORCE_GT(eps_, 0.0);
        CAFFE_ENFORCE_LT(eps_, 0.5);
        */
    }
}

///---------------------------------------
pub struct LogitGradientOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    eps: f32,

    phantom: PhantomData<T>,
}

num_inputs!{LogitGradient, 2}

num_outputs!{LogitGradient, 1}

inputs!{LogitGradient, 
    0 => ("X", "input float tensor"),
    1 => ("dY", "input float tensor")
}

outputs!{LogitGradient, 
    0 => ("dX", "output float tensor")
}

args!{LogitGradient, 
    0 => ("eps", "small positive epsilon value, the default is 1e-6.")
}


impl<T,Context> LogitGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            eps_(this->template GetSingleArgument<float>("eps", 1e-6f))
        */
    }
}

///--------------------------------------
impl LogitFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self, 
        size:     i32,
        x:        *const T,
        y:        *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ConstEigenVectorMap<T> X_vec(X, size);
          EigenVectorMap<T> Y_vec(Y, size);
          Y_vec = X_vec.array().min(static_cast<T>(1.0f - eps_));
          Y_vec = Y_vec.array().max(eps_);
          Y_vec = (Y_vec.array() / (T(1) - Y_vec.array())).log();
          return true;
        */
    }
}

impl LogitGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& dY = Input(1);

      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      int channels = X.dim32(X.dim() - 1);
      ConstEigenArrayMap<float> Xmat(
          X.template data<float>(), channels, X.numel() / channels);
      ConstEigenArrayMap<float> dYmat(
          dY.template data<float>(), channels, X.numel() / channels);
      EigenArrayMap<float> dXmat(
          dX->template mutable_data<float>(), channels, X.numel() / channels);
      dXmat = (Xmat < eps_ || Xmat > 1.0 - eps_)
                  .select(0, dYmat * ((1 - Xmat) * Xmat).inverse());
      return true;
        */
    }
}

register_cpu_operator!{
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        LogitFunctor<CPUContext>>
}

register_cpu_operator!{LogitGradient, LogitGradientOp<f32, CPUContext>}

pub struct GetLogitGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLogitGradient<'a> {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>{CreateOperatorDef(
            "LogitGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)})};
        */
    }
}

register_gradient!{Logit, GetLogitGradient}

pub type LogitOp = UnaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    CPUContext,
    LogitFunctor<CPUContext>>;

