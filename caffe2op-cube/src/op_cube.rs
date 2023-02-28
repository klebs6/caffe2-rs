crate::ix!();

use crate::{
    OperatorDef,
    GradientMakerBase,
    CPUContext,
};

pub struct CubeFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> CubeFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cube<T, Context>(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Cube,
    UnaryElementwiseOp<NumericTypes, CPUContext, CubeFunctor<CPUContext>>
}

num_inputs!{Cube, 1}

num_outputs!{Cube, 1}

inputs!{Cube, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Cube, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cube of the input tensor, element-wise.")
}

identical_type_and_shape!{Cube}

///------------------------------------------------------

pub struct CubeGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

register_cpu_operator!{
    CubeGradient,
    BinaryElementwiseOp<
        NumericTypes,
        CPUContext,
        CubeGradientFunctor<CPUContext>>
}

num_inputs!{CubeGradient, 2}

num_outputs!{CubeGradient, 1}

identical_type_and_shape!{CubeGradient}

impl CubeGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        dY_dims:   &Vec<i32>,
        x_dims:    &Vec<i32>,
        dY:        *const T,
        x:         *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
          dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
      EigenVectorMap<T>(dX, size) = ConstEigenVectorArrayMap<T>(dY, size) *
          ConstEigenVectorArrayMap<T>(X, size).square() * T(3);
      return true;
        */
    }
}

pub struct GetCubeGradient;

impl GetGradientDefs for GetCubeGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CubeGradient",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cube, GetCubeGradient}
