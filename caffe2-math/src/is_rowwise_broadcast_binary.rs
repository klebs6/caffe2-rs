crate::ix!();

#[inline] pub fn is_rowwise_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    rows:          *mut i32,
    cols:          *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pivot = 0;
      for (; A_pivot < ndim && A_dims[A_pivot] == 1; ++A_pivot)
        ;
      int B_pivot = 0;
      for (; B_pivot < ndim && B_dims[B_pivot] == 1; ++B_pivot)
        ;
      if (A_pivot == B_pivot) {
        return false;
      }
      const int pivot = std::max(A_pivot, B_pivot);
      if (A_pivot > B_pivot) {
        *rows = c10::multiply_integers(B_dims + B_pivot, B_dims + pivot);
        *broadcast_1st = true;
      } else {
        *rows = c10::multiply_integers(A_dims + A_pivot, A_dims + pivot);
        *broadcast_1st = false;
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

#[inline] pub fn is_colwise_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    rows:          *mut i32,
    cols:          *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pivot = ndim - 1;
      for (; A_pivot >= 0 && A_dims[A_pivot] == 1; --A_pivot)
        ;
      int B_pivot = ndim - 1;
      for (; B_pivot >= 0 && B_dims[B_pivot] == 1; --B_pivot)
        ;
      if (A_pivot == B_pivot) {
        return false;
      }
      ++A_pivot;
      ++B_pivot;
      const int pivot = std::min(A_pivot, B_pivot);
      if (A_pivot < B_pivot) {
        *cols = c10::multiply_integers(B_dims + pivot, B_dims + B_pivot);
        *broadcast_1st = true;
      } else {
        *cols = c10::multiply_integers(A_dims + pivot, A_dims + A_pivot);
        *broadcast_1st = false;
      }
      *rows = 1;
      for (int i = 0; i < pivot; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *rows *= A_dims[i];
      }
      return true;
    */
}
