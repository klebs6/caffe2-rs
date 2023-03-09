crate::ix!();

pub struct TanGradientFunctor<Context> { 

    phantom: PhantomData<Context>,
}

impl<CPUContext> TanGradientFunctor<CPUContext> {

    pub fn forward<T>(
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
           EigenVectorMap<T>(dX, size) = dY_arr / X_arr.cos().square();
           return true;
           */
    }
}

register_cpu_operator!{
    TanGradient,
    BinaryElementwiseOp<TensorTypes<f32>, CPUContext, TanGradientFunctor<CPUContext>>
}

num_inputs!{TanGradient, 2}

num_outputs!{TanGradient, 1}

identical_type_and_shape!{TanGradient}
