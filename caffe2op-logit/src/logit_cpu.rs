crate::ix!();

impl LogitFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self, 
        size:     i32,
        x:        *const T,
        y:        *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ConstEigenVectorMap<T> X_vec(X, size);
          EigenVectorMap<T> Y_vec(Y, size);
          Y_vec = X_vec.array().min(static_cast<T>(1.0f - eps_));
          Y_vec = Y_vec.array().max(eps_);
          Y_vec = (Y_vec.array() / (T(1) - Y_vec.array())).log();
          return true;
        */
    }
}
