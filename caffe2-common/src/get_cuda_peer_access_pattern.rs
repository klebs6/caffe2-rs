crate::ix!();

/**
  | Return a peer access pattern by returning
  | a matrix (in the format of a nested vector)
  | of boolean values specifying whether
  | peer access is possible.
  | 
  | This function returns false if anything
  | wrong happens during the query of the
  | GPU access pattern.
  |
  */
#[inline] pub fn get_cuda_peer_access_pattern(pattern: *mut Vec<Vec<bool>>) -> bool {
    
    todo!();
    /*
        int gpu_count;
      if (cudaGetDeviceCount(&gpu_count) != cudaSuccess) return false;
      pattern->clear();
      pattern->resize(gpu_count, vector<bool>(gpu_count, false));
      for (int i = 0; i < gpu_count; ++i) {
        for (int j = 0; j < gpu_count; ++j) {
          int can_access = true;
          if (i != j) {
            if (cudaDeviceCanAccessPeer(&can_access, i, j)
                     != cudaSuccess) {
              return false;
            }
          }
          (*pattern)[i][j] = static_cast<bool>(can_access);
        }
      }
      return true;
    */
}
