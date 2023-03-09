crate::ix!();

register_cpu_operator!{Swish,
    UnaryElementwiseOp<
    TensorTypes<f32>,
    CPUContext,
    SwishFunctor<CPUContext>>
}

impl<T, CPUContext> SwishFunctor<T, CPUContext> {

    #[inline] pub fn invoke(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
                EigenVectorArrayMap<T>(Y, N) = X_arr / (T(1) + (-X_arr).exp());
                return true;
        */
    }
}
