crate::ix!();

use crate::{
    GradientMakerBase,
    CPUContext,
    OperatorDef
};

/**
  | Calculates the arcsine of the given
  | input tensor, element-wise.
  |
  */
pub struct AsinFunctor<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{Asin, 1}

num_outputs!{Asin, 1}

inputs!{Asin, 
    0 => ("input", "Input tensor")
}

outputs!{Asin, 
    0 => ("output", "The arcsine of the input tensor computed element-wise")
}

identical_type_and_shape!{Asin}

impl<Context> AsinFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Asin(N, X, Y, context);
            return true;
        */
    }
}

pub struct AsinGradientFunctor<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{AsinGradient, 2}

num_outputs!{AsinGradient, 1}

identical_type_and_shape!{AsinGradient}

impl<Context> AsinGradientFunctor<Context> {

    #[inline] pub fn forward<T>(
        &mut self,
        x_dims:   &Vec<i32>,
        dY_dims:  &Vec<i32>,
        x:        *const T,
        dY:       *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = dY_arr * (T(1) - X_arr.square()).rsqrt();
          return true;
        */
    }
}

register_cpu_operator!{
    Asin,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AsinFunctor<CPUContext>>
}

register_cpu_operator!{
    AsinGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AsinGradientFunctor<CPUContext>>
}

pub struct GetAsinGradient;

impl GetGradientDefs for GetAsinGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AsinGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Asin, GetAsinGradient}
