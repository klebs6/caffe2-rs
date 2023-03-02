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

