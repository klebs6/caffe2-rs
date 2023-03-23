crate::ix!();

#[inline] pub fn is_batch_transpose2D(ndim: i32, axes: *const i32) -> bool {
    
    todo!();
    /*
        if (ndim < 2) {
        return false;
      }
      for (int i = 0; i < ndim - 2; ++i) {
        if (axes[i] != i) {
          return false;
        }
      }
      return axes[ndim - 2] == ndim - 1 && axes[ndim - 1] == ndim - 2;
    */
}
