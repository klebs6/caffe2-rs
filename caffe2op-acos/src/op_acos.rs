crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    CPUContext
};

/**
  | Calculates the arccosine of the given
  | input tensor, element-wise.
  |
  */
pub struct AcosFunctor<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{Acos, 1}

num_outputs!{Acos, 1}

inputs!{Acos, 
    0 => ("input", "Input tensor")
}

outputs!{Acos, 
    0 => ("output", "The arccosine of the input tensor computed element-wise")
}

identical_type_and_shape!{Acos}

register_cpu_operator!{
    Acos,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AcosFunctor<CPUContext>>
}

impl<Context> AcosFunctor<Context> {

    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Acos(N, X, Y, context);
            return true;
        */
    }
}

pub struct AcosGradientFunctor<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{AcosGradient, 2}
num_outputs!{AcosGradient, 1}
identical_type_and_shape!{AcosGradient}

register_cpu_operator!{
    AcosGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AcosGradientFunctor<CPUContext>>
}

impl AcosGradientFunctor<CPUContext> {

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
          EigenVectorMap<T>(dX, size) = -dY_arr * (T(1) - X_arr.square()).rsqrt();
          return true;
        */
    }
}

pub struct GetAcosGradient;

impl GetGradientDefs for GetAcosGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AcosGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Acos, GetAcosGradient}
