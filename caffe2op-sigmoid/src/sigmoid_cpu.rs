crate::ix!();

impl SigmoidFunctor<CPUContext> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
          EigenVectorArrayMap<T>(Y, N) =
          T(1) / (T(1) + (-ConstEigenVectorArrayMap<T>(X, N)).exp());
          return true;
        */
    }
}
