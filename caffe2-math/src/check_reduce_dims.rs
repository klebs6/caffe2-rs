crate::ix!();

#[inline] pub fn check_reduce_dims(
    ndim:   i32,
    x_dims: *const i32,
    y_dims: *const i32) -> bool {
    
    todo!();
    /*
        for (int i = 0; i < ndim; ++i) {
        if (X_dims[i] != Y_dims[i] && Y_dims[i] != 1) {
          return false;
        }
      }
      return true;
    */
}
