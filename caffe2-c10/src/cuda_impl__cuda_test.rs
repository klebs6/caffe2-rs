/*!
  | Just a little test file to make sure that
  | the Cuda library works
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/impl/CUDATest.h]
//-------------------------------------------[.cpp/pytorch/c10/cuda/impl/CUDATest.cpp]

pub fn c10_has_cuda_gpu() -> bool {
    
    todo!();
        /*
            int count;
      C10_CUDA_CHECK(cudaGetDeviceCount(&count));

      return count != 0;
        */
}

pub fn c10_cuda_test() -> i32 {
    
    todo!();
        /*
            int r = 0;
      if (has_cuda_gpu()) {
        C10_CUDA_CHECK(cudaGetDevice(&r));
      }
      return r;
        */
}

/// This function is not exported
pub fn c10_cuda_private_test() -> i32 {
    
    todo!();
        /*
            return 2;
        */
}
