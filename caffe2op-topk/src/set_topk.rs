crate::ix!();

#[inline] pub fn set_top_kgradient<T>(
    values:      *const T,
    indices:     *const i64,
    k:           i32,
    src_offset:  i64,
    dst_offset:  i64,
    stride:      i64,
    gradient:    *mut T) 
{
    todo!();
    /*
        int64_t src_pos = src_offset;
      for (int i = 0; i < k; ++i) {
        if (indices[src_pos] < 0) {
          continue;
        }
        gradient[dst_offset + indices[src_pos] * stride] = values[src_pos];
        src_pos += stride;
      }
    */
}
