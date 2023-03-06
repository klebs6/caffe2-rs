crate::ix!();

/**
  | Returns the indices that would sort
  | an array. For example:
  | 
  | data = [3, 1, 2, 4]
  | 
  | return = [1, 2, 0, 3] (reverse = false)
  | 
  | return = [3, 0, 2, 1] (reverse = true)
  |
  */
#[inline] pub fn arg_sort<TDATA, TIDX>(
    data:    *const TDATA,
    idx:     *mut TIDX,
    n:       usize,
    reverse: bool)  {
    todo!();
    /*
        std::function<bool(size_t, size_t)> cmp_lambda;
      if (reverse) {
        cmp_lambda = [data](size_t i, size_t j) { return data[i] > data[j]; };
      } else {
        cmp_lambda = [data](size_t i, size_t j) { return data[i] < data[j]; };
      }
      size_t n = 0;
      std::generate(idx, idx + N, [&n] { return n++; });
      std::sort(idx, idx + N, cmp_lambda);
    */
}
