crate::ix!();

pub struct CosGradientFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{CosGradient, 2}

num_outputs!{CosGradient, 1}

identical_type_and_shape!{CosGradient}

register_cpu_operator!{
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CosGradientFunctor<CPUContext>>
}

impl CosGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        x_dims:  &Vec<i32>,
        dY_dims: &Vec<i32>,
        x:       *const T,
        dY:      *const T,
        dX:      *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = -dY_arr * X_arr.sin();
          return true;
        */
    }
}
