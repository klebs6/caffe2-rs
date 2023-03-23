crate::ix!();

#[inline] pub fn is_both_ends_broadcast_binary_op(
    ndim:          i32,
    a_dims:        *const i32,
    b_dims:        *const i32,
    pre:           *mut i32,
    mid:           *mut i32,
    nxt:           *mut i32,
    broadcast_1st: *mut bool) -> bool {
    
    todo!();
    /*
        if (ndim == 0) {
        return false;
      }
      int A_pre = 0;
      for (; A_pre < ndim && A_dims[A_pre] == 1; ++A_pre)
        ;
      int B_pre = 0;
      for (; B_pre < ndim && B_dims[B_pre] == 1; ++B_pre)
        ;
      int A_nxt = ndim - 1;
      for (; A_nxt >= 0 && A_dims[A_nxt] == 1; --A_nxt)
        ;
      int B_nxt = ndim - 1;
      for (; B_nxt >= 0 && B_dims[B_nxt] == 1; --B_nxt)
        ;
      ++A_nxt;
      ++B_nxt;
      if (A_pre == B_pre || A_nxt == B_nxt) {
        return false;
      }
      if (A_pre > B_pre && A_nxt < B_nxt) {
        *pre = c10::multiply_integers(B_dims + B_pre, B_dims + A_pre);
        *nxt = c10::multiply_integers(B_dims + A_nxt, B_dims + B_nxt);
        *broadcast_1st = true;
      } else if (A_pre < B_pre && A_nxt > B_nxt) {
        *pre = c10::multiply_integers(A_dims + A_pre, A_dims + B_pre);
        *nxt = c10::multiply_integers(A_dims + B_nxt, A_dims + A_nxt);
        *broadcast_1st = false;
      } else {
        return false;
      }
      const int l = std::max(A_pre, B_pre);
      const int r = std::min(A_nxt, B_nxt);
      *mid = 1;
      for (int i = l; i < r; ++i) {
        if (A_dims[i] != B_dims[i]) {
          return false;
        }
        *mid *= A_dims[i];
      }
      return true;
    */
}
