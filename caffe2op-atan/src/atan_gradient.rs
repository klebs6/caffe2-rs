crate::ix!();

pub struct AtanGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{AtanGradient, 2}

num_outputs!{AtanGradient, 1}

identical_type_and_shape!{AtanGradient}

register_cpu_operator!{
    AtanGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AtanGradientFunctor<CPUContext>>
}

impl AtanGradientFunctor<CPUContext> {

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
          EigenVectorMap<T>(dX, size) = dY_arr / (T(1) + X_arr.square());
          return true;
        */
    }
}
