crate::ix!();

#[inline] pub fn batch_permutation_loop<const forwards: bool>(
    n:        i32,
    k:        i32,
    src:      *const f32,
    indices:  *const i32,
    dst:      *mut f32) 
{
    todo!();
    /*
        long numBytes = K * sizeof(float);
      if (forwards) {
    #ifdef _OPENMP
    #if (_OPENMP >= 201307)
    #pragma omp parallel for simd
    #else
    #pragma omp parallel for
    #endif
    #endif
        for (int n = 0; n < N; n++) {
          int origIdx = n * K;
          int permuteIdx = indices[n] * K;
          std::memcpy(dst + origIdx, src + permuteIdx, numBytes);
        }
      } else {
        std::vector<int> backward_indices(N);
        for (int i = 0; i < N; ++i) {
          backward_indices[indices[i]] = i;
        }
        for (int n = 0; n < N; n++) {
          int permuteIdx = n * K;
          int origIdx = backward_indices[n] * K;
          std::memcpy(dst + permuteIdx, src + origIdx, numBytes);
        }
      }
    */
}
