crate::ix!();

impl SoftsignFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
      EigenVectorMap<T>(Y, N) = (T(1) + X_arr.abs()).inverse() * X_arr;
      return true;
        */
    }
}
