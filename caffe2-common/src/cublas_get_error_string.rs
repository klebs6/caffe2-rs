crate::ix!();

/**
  | Return a human readable cublas error
  | string.
  |
  */
#[inline] pub fn cublas_get_error_string(error: CuBlasStatus) -> *const u8 {
    
    todo!();
    /*
        switch (error) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    #ifndef __HIP_PLATFORM_HCC__
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    #else
      case rocblas_status_invalid_size:
        return "rocblas_status_invalid_size";
      case rocblas_status_perf_degraded:
        return "rocblas_status_perf_degraded";
      case rocblas_status_size_query_mismatch:
        return "rocblas_status_size_query_mismatch";
      case rocblas_status_size_increased:
        return "rocblas_status_size_increased";
      case rocblas_status_size_unchanged:
        return "rocblas_status_size_unchanged";
    #endif
      }
      // To suppress compiler warning.
      return "Unrecognized cublas error string";
    */
}
