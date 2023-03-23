crate::ix!();

#[inline] pub fn compute_transpose_axes_for_reduce_op_with_reduce_axes(
    num_dims:        i32,
    num_reduce_axes: i32,
    reduce_axes:     *const i32,
    transpose_axes:  *mut i32)  {
    
    todo!();
    /*
        const int d = num_dims - num_reduce_axes;
      std::copy_n(reduce_axes, num_reduce_axes, transpose_axes + d);
      std::sort(transpose_axes + d, transpose_axes + num_dims);
      int p = 0;
      int q = d;
      for (int i = 0; i < num_dims; ++i) {
        if (q < num_dims && i == transpose_axes[q]) {
          ++q;
        } else {
          transpose_axes[p++] = i;
        }
      }
    */
}
