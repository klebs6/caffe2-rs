crate::ix!();

/**
  | Returns the number of devices.
  |
  */
#[inline] pub fn num_cuda_devices() -> i32 {
    
    todo!();
    /*
        if (getenv("CAFFE2_DEBUG_CUDA_INIT_ORDER")) {
        static bool first = true;
        if (first) {
          first = false;
          std::cerr << "DEBUG: caffe2::NumCudaDevices() invoked for the first time"
                    << std::endl;
        }
      }
      // It logs warnings on first run
      return CudaDeviceCount();
    */
}
