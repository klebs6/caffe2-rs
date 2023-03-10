crate::ix!();

pub struct AbsGradientFunctor<Context> {
    context: Context,
}

impl AbsGradientFunctor<CPUContext> {

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
          EigenVectorMap<T>(dX, size) =
              (X_arr == T(0)).select(T(0), (X_arr > T(0)).select(dY_arr, -dY_arr));
          return true;
        */
    }
}

register_cpu_operator!{
    Abs,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, AbsFunctor<CPUContext>>
}

register_cpu_operator!{
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AbsGradientFunctor<CPUContext>>
}

num_inputs!{AbsGradient, 2}

num_outputs!{AbsGradient, 1}

identical_type_and_shape_of_input!{AbsGradient, 0}

