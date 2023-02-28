crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
    CPUContext
};

/**
  | Mish takes one input data (Tensor) and
  | produces one output data (Tensor) where
  | the Mish function, y = x * tanh(ln(1 +
  | exp(x))), is applied to the tensor elementwise.
  |
  */
pub struct MishFunctor<Context> { 
    // Input: X, output: Y
    phantom: PhantomData<Context>,
}

num_inputs!{Mish, 1}

num_outputs!{Mish, 1}

inputs!{Mish, 
    0 => ("X", "1D input tensor")
}

outputs!{Mish, 
    0 => ("Y", "1D output tensor")
}

identical_type_and_shape!{Mish}

impl MishFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
        todo!();
        /*
          ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorArrayMap<T> Y_arr(Y, N);
          math::Exp<T, CPUContext>(N, X, Y, context);
          math::Log1p<T, CPUContext>(N, Y, Y, context);
          Y_arr = X_arr * Y_arr.tanh();
          return true;
        */
    }
}

/**
  | MishGradient takes X, Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the Mish function.
  |
  */
pub struct MishGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

/**
  | Input: X, Y, dY,
  | 
  | output: dX
  |
  */
num_inputs!{MishGradient, 3}

num_outputs!{MishGradient, 1}

impl<Context> MishGradientOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(INPUT));
        */
    }
}

input_tags!{
    MishGradientOp {
        Input,
        Output,
        OutputGrad
    }
}

output_tags!{
    MishGradientOp {
        InputGrad
    }
}

impl MishGradientOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& Y = Input(OUTPUT);
      const auto& dY = Input(OUTPUT_GRAD);

      CAFFE_ENFORCE_EQ(X.numel(), Y.numel());
      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());

      const T* X_data = X.template data<T>();
      const T* Y_data = Y.template data<T>();
      const T* dY_data = dY.template data<T>();
      T* dX_data = dX->template mutable_data<T>();

      const int64_t N = X.numel();
      ConstEigenVectorArrayMap<T> X_arr(X_data, N);
      ConstEigenVectorArrayMap<T> Y_arr(Y_data, N);
      ConstEigenVectorArrayMap<T> dY_arr(dY_data, N);
      EigenVectorArrayMap<T> dX_arr(dX_data, N);

      math::Exp<T, CPUContext>(N, X_data, dX_data, &context_);
      math::Log1p<T, CPUContext>(N, dX_data, dX_data, &context_);
      math::Tanh<T, CPUContext>(N, dX_data, dX_data, &context_);
      dX_arr = dY_arr *
          (dX_arr +
           X_arr * (T(1) - dX_arr.square()) * T(0.5) *
               ((X_arr * T(0.5)).tanh() + T(1)));

      return true;
        */
    }
}

register_cpu_operator!{
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        MishFunctor<CPUContext>>
}

register_cpu_operator!{MishGradient, MishGradientOp<CPUContext>}

pub struct GetMishGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMishGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MishGradient",
            "",
            std::vector<std::string>{I(0), O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Mish, GetMishGradient}
