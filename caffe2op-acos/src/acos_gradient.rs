crate::ix!();

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
