crate::ix!();

/**
  | Gets the current GPU id. This is a simple
  | wrapper around cudaGetDevice().
  |
  */
#[inline] pub fn caffe_cuda_get_device() -> i32 {
    
    todo!();
    /*
        int gpu_id = 0;
      CUDA_ENFORCE(cudaGetDevice(&gpu_id));
      return gpu_id;
    */
}

/**
  | Gets the current GPU id. This is a simple
  | wrapper around cudaGetDevice().
  |
  */
#[inline] pub fn caffe_cuda_set_device(id: i32)  {
    
    todo!();
    /*
        CUDA_ENFORCE(cudaSetDevice(id));
    */
}
