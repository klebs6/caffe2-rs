crate::ix!();

#[inline] pub fn is_both_ends_reduce(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    pre:    *mut i32,
    mid:    *mut i32,
    nxt:    *mut i32) -> bool {
    
    todo!();
    /*
        *nxt = 1;
      int r = ndim - 1;
      for (; r >= 0 && B_dims[r] == 1; --r) {
        *nxt *= A_dims[r];
      }
      *pre = 1;
      int l = 0;
      for (; l <= r && B_dims[l] == 1; ++l) {
        *pre *= A_dims[l];
      }
      *mid = 1;
      for (int i = l; i <= r; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *mid *= A_dims[i];
      }
      return true;
    */
}
