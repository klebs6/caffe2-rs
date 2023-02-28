crate::ix!();

use crate::{
    CPUContext,
    OperatorDef,
    GradientMakerBase
};

pub struct CbrtFunctor<Context> {
    phantom: PhantomData<Context>,
}

register_cpu_operator!{
    Cbrt,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CbrtFunctor<CPUContext>>
}

num_inputs!{Cbrt, 1}

num_outputs!{Cbrt, 1}

inputs!{Cbrt, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
} 

outputs!{Cbrt, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cbrt of the input tensor, element-wise.")
} 

allow_inplace!{Cbrt, vec![(0, 0)]}

identical_type_and_shape!{Cbrt} 

impl<Context> CbrtFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self, 
        n: i32,
        x: *const T,
        y: *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cbrt<T, Context>(N, X, Y, context);
            return true;
        */
    }
}

pub struct CbrtGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{CbrtGradient, 2}

num_outputs!{CbrtGradient, 1}

register_cpu_operator!{
    CbrtGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CbrtGradientFunctor<CPUContext>>
}

allow_inplace!{CbrtGradient, vec![(0, 0)]}

identical_type_and_shape_of_input!{CbrtGradient, 0}

impl CbrtGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self, 
        dY_dims:  &Vec<i32>,
        y_dims:   &Vec<i32>,
        dY:       *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
          EigenVectorMap<T>(dX, size) = ConstEigenVectorArrayMap<T>(dY, size) /
              ConstEigenVectorArrayMap<T>(Y, size).square() / T(3);
          return true;
        */
    }
}

pub struct GetCbrtGradient;

impl GetGradientDefs for GetCbrtGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CbrtGradient",
            "",
            std::vector<std::string>{GO(0), O(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cbrt, GetCbrtGradient}
