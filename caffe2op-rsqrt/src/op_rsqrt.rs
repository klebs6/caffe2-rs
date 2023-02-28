crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    CPUContext
};

/**
  | Computes the element-wise rsqrt of
  | the input.
  |
  */
pub struct RsqrtFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

num_inputs!{Rsqrt, 1}

num_outputs!{Rsqrt, 1}

inputs!{Rsqrt, 
    0 => ("X", "ND input tensor")
}

outputs!{Rsqrt, 
    0 => ("Y", "ND output tensor")
}

identical_type_and_shape!{Rsqrt}

allow_inplace!{Rsqrt, vec![(0, 0)]}

impl<Context> RsqrtFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Rsqrt<T, Context>(N, X, Y, context);
        return true;
        */
    }
}

///-------------
pub struct RsqrtGradientFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

num_inputs!{RsqrtGradient, 2}

num_outputs!{RsqrtGradient, 1}

allow_inplace!{RsqrtGradient, vec![(0, 0)]}

impl RsqrtGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        dy_dims: &Vec<i32>,
        y_dims:  &Vec<i32>,
        dy:      *const T,
        y:       *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
      EigenVectorMap<T>(dX, size) = ConstEigenVectorMap<T>(dY, size).array() *
          ConstEigenVectorMap<T>(Y, size).array().cube() * static_cast<T>(-0.5);
      return true;
        */
    }
}

register_cpu_operator!{Rsqrt,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        RsqrtFunctor<CPUContext>>}

register_cpu_operator!{RsqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        RsqrtGradientFunctor<CPUContext>>}

pub struct GetRsqrtGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRsqrtGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RsqrtGradient",
            "",
            std::vector<std::string>{GO(0), O(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Rsqrt, GetRsqrtGradient}
