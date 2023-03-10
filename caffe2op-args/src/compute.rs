crate::ix!();

#[inline] pub fn compute_arg_impl<T, Compare, Context>(
    prev_size:  i32,
    next_size:  i32,
    n:          i32,
    comp:       &Compare,
    x:          *const T,
    y:          *mut i64,
    context:    *mut Context) 
{
    todo!();
    /*
        math::Set<int64_t, Context>(prev_size * next_size, int64_t(0), Y, context);
      for (int i = 0; i < prev_size; ++i) {
        const T* cur_X = X + i * n * next_size + next_size;
        for (int k = 1; k < n; ++k) {
          for (int j = 0; j < next_size; ++j) {
            int64_t* cur_Y = Y + i * next_size + j;
            if (comp(*cur_X, X[i * n * next_size + *cur_Y * next_size + j])) {
              *cur_Y = k;
            }
            ++cur_X;
          }
        }
      }
    */
}
