crate::ix!();

#[inline] pub fn is_rowwise_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    rows:   *mut i32,
    cols:   *mut i32) -> bool {
    
    todo!();
    /*
        *cols = 1;
      int pivot = ndim - 1;
      for (; pivot >= 0 && B_dims[pivot] == 1; --pivot) {
        *cols *= A_dims[pivot];
      }
      *rows = 1;
      for (int i = pivot; i >= 0; --i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *rows *= A_dims[i];
      }
      return true;
    */
}

#[inline] pub fn is_colwise_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    rows:   *mut i32,
    cols:   *mut i32) -> bool {
    
    todo!();
    /*
        *rows = 1;
      int pivot = 0;
      for (; pivot < ndim && B_dims[pivot] == 1; ++pivot) {
        *rows *= A_dims[pivot];
      }
      *cols = 1;
      for (int i = pivot; i < ndim; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *cols *= A_dims[i];
      }
      return true;
    */
}
