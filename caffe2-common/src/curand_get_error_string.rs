crate::ix!();

/**
  | Return a human readable curand error
  | string.
  |
  */
#[inline] pub fn curand_get_error_string(error: CuRandStatus) -> *const u8 {
    
    todo!();
    /*
        switch (error) {
      case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
      case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
      case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
      case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
      case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    #ifdef __HIP_PLATFORM_HCC__
      case HIPRAND_STATUS_NOT_IMPLEMENTED:
        return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    #endif
      }
      // To suppress compiler warning.
      return "Unrecognized curand error string";
    */
}
