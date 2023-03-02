crate::ix!();

/// Helpers for copying parameters.
#[inline] pub fn tile_array_into_vector<T>(
    a: *const T,
    d: i32,
    k: i32,
    b: *mut Vec<T>) 
{
    todo!();
    /*
        b->resize(K * D);
      for (int k = 0; k < K; k++) {
        std::copy(a, a + D, b->begin() + k * D);
      }
    */
}

#[inline] pub fn tile_indices_in_place(
    v: *mut Vec<i32>,
    d: i32,
    k: i32)  
{
    todo!();
    /*
        int n = v->size();
      v->resize(K * n);
      for (int k = 1; k < K; k++) {
        for (int j = 0; j < n; j++) {
          (*v)[k * n + j] = (*v)[j] + k * D;
        }
      }
    */
}
