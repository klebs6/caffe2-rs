crate::ix!();

impl ReplaceNaNOp<CPUContext> {

    #[inline] pub fn replace_nan<T>(&mut self, 
        value: &T,
        size:  i64,
        x:     *const T,
        y:     *mut T)  {
    
        todo!();
        /*
            for (int64_t i = 0; i < size; i++) {
        if (std::isnan(X[i])) {
          Y[i] = value;
        } else {
          Y[i] = X[i];
        }
      }
        */
    }
}
