crate::ix!();

#[inline] pub fn compute_transpose_axes_for_reduce_op(
    ndim: i32,
    dims: *const i32,
    axes: *mut i32)  {
    
    todo!();
    /*
        const int d = ndim - std::count(dims, dims + ndim, 1);
      int p = 0;
      int q = d;
      for (int i = 0; i < ndim; ++i) {
        if (dims[i] == 1) {
          axes[q++] = i;
        } else {
          axes[p++] = i;
        }
      }
    */
}
