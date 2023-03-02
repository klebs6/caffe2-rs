crate::ix!();

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
