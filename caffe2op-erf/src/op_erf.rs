crate::ix!();

pub struct ErfFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> ErfFunctor<Context> {

    #[inline] pub fn call<T>(
        &self,
        n:        i32,
        x:        *const T,
        y:        *mut T,
        context:  *mut Context) -> bool 
    {
        todo!();
        /*
            math::Erf(N, X, Y, context);
            return true;
        */
    }
}

pub struct ErfGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl ErfGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self,
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
          EigenVectorMap<T>(dX, size) = T(2) / sqrtf(PI) * (-X_arr.square()).exp() * dY_arr;
          return true;
        */
    }
}

register_cpu_operator!{
    Erf,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        ErfFunctor<CPUContext>>
}

register_cpu_operator!{
    ErfGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        ErfGradientFunctor<CPUContext>>
}

/**
  | Calculates the arcsine of the given
  | input tensor, element-wise.
  |
  */
pub struct Erf {

}

num_inputs!{Erf, 1}

num_outputs!{Erf, 1}

inputs!{Erf, 
    0 => ("input", "Input tensor")
}

outputs!{Erf, 
    0 => ("output", "The arcsine of the input tensor computed element-wise")
}

identical_type_and_shape!{Erf}

pub struct ErfGradient {

}

num_inputs!{ErfGradient, 2}

num_outputs!{ErfGradient, 1}

identical_type_and_shape!{ErfGradient}

pub struct GetErfGradient;

impl GetGradientDefs for GetErfGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ErfGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Erf, GetErfGradient}
