crate::ix!();

/**
  | A helper function to obtain cudnn error
  | strings.
  |
  */
#[inline] pub fn cudnn_get_error_string(status: CudnnStatus) -> *const u8 {
    
    todo!();
    /*
        switch (status) {
        case CUDNN_STATUS_SUCCESS:
          return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED:
          return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED:
          return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM:
          return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR:
          return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE:
          return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH:
          return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR:
          return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED:
          return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED:
          return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR:
          return "CUDNN_STATUS_LICENSE_ERROR";
        default:
          return "Unknown cudnn error number";
      }
    */
}
