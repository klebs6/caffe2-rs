crate::ix!();

use crate::{
    SameTypeAsInput,
    BinaryElementwiseWithArgsOp,
    GradientMakerBase,
    OperatorDef,
    TensorTypes,
    TensorShape,
    UnaryElementwiseWithArgsOp,
    CPUContext,
    OpSchemaCost,
    OperatorStorage,
};

pub const kFastCoeff: f32 = 0.044715;

///--------------------------------
pub struct GeluFunctor<Context> {
    fast_gelu: bool,

    phantom: PhantomData<Context>,
}

impl<Context> GeluFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false))
        */
    }
}

impl GeluFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n:        i32,
        x:        *const T,
        y:        *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            if (fast_gelu) {
            // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
            ConstEigenVectorArrayMap<T> X_arr(X, N);
            EigenVectorArrayMap<T> Y_arr(Y, N);
            Y_arr = X_arr *
                (((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh() +
                 T(1)) *
                static_cast<T>(0.5);
          } else {
            // y = x * P(X <= x) where X ~ N(0, 1)
            math::CdfNorm<T, CPUContext>(N, X, Y, context);
            math::Mul<T, CPUContext>(N, X, Y, Y, context);
          }
          return true;
        */
    }
}

///--------------------------------
pub struct GeluGradientFunctor<Context> {
    fast_gelu: bool,

    phantom: PhantomData<Context>,
}

impl<Context> GeluGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false))
        */
    }
}

impl GeluGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self,
        dY_dims:   &Vec<i32>,
        x_dims:    &Vec<i32>,
        dY:        *const T,
        x:         *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int N = std::accumulate(
              dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, N);
          ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorArrayMap<T> dX_arr(dX, N);
          if (fast_gelu) {
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;
            constexpr T kBeta = kAlpha * gelu_utils::kFastCoeff * T(3);
            dX_arr = ((X_arr + X_arr.cube() * gelu_utils::kFastCoeff) * kAlpha).tanh();
            dX_arr =
                (T(1) + dX_arr +
                 X_arr * (T(1) - dX_arr.square()) * (kBeta * X_arr.square() + kAlpha)) *
                dY_arr * static_cast<T>(0.5);
          } else {
            constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
            math::CdfNorm<T, CPUContext>(N, X, dX, context);
            dX_arr = (dX_arr +
                      X_arr * (-X_arr.square() * static_cast<T>(0.5)).exp() * kAlpha) *
                dY_arr;
          }
          return true;
        */
    }
}

/**
  | Relu takes one input data (Tensor) and
  | produces one output data (Tensor) where
  | the rectified linear function, y = xP(X
  | <= x) where X ~ N(0, 1), is applied to the
  | tensor elementwise.
  | 
  | Input: X, output: Y
  |
  */
pub type GeluOp<Context> = UnaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    Context,
    GeluFunctor<Context>>;

num_inputs!{Gelu, 1}

num_outputs!{Gelu, 1}

args!{Gelu, 
    0 => ("fast_gelu", "If true, use y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3))).")
}

cost_inference_function!{Gelu, CostInferenceForGelu}

identical_type_and_shape!{Gelu}

inputs!{Gelu, 
    0 => ("X", "1D input tensor")
}

outputs!{Gelu, 
    0 => ("Y", "1D input tensor")
}

register_cpu_operator!{Gelu,         GeluOp<CPUContext>}

///--------------------------------
pub type GeluGradientOp<Context> = BinaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    Context,
    GeluGradientFunctor<Context>,
    SameTypeAsInput>;

num_inputs!{GeluGradient, 2}

num_outputs!{GeluGradient, 1}

identical_type_and_shape_of_input!{GeluGradient, 1}

register_cpu_operator!{GeluGradient, GeluGradientOp<CPUContext>}

#[inline] pub fn cost_inference_for_gelu(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

///---------------------------
pub struct GetGeluGradient;

impl GetGradientDefs for GetGeluGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "GeluGradient",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Gelu, GetGeluGradient}
