crate::ix!();

impl<CPUContext> EluFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorMap<T>(Y, N) =
              (X_arr < 0).select(alpha * (X_arr.exp() - T(1)), X_arr);
          return true;
        */
    }
}
