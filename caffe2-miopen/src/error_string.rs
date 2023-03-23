crate::ix!();

/**
  | A helper function to obtain miopen error
  | strings.
  |
  */
#[inline] pub fn miopen_get_error_string(status: miopenStatus_t) -> *const u8 {
    
    todo!();
    /*
        switch(status)
        {
        case miopenStatusSuccess: return "MIOPEN_STATUS_SUCCESS";
        case miopenStatusNotInitialized: return "MIOPEN_STATUS_NOT_INITIALIZED";
        case miopenStatusAllocFailed: return "MIOPEN_STATUS_ALLOC_FAILED";
        case miopenStatusBadParm: return "MIOPEN_STATUS_BAD_PARAM";
        case miopenStatusInternalError: return "MIOPEN_STATUS_INTERNAL_ERROR";
        case miopenStatusInvalidValue: return "MIOPEN_STATUS_INVALID_VALUE";
        case miopenStatusNotImplemented: return "MIOPEN_STATUS_NOT_SUPPORTED";
        case miopenStatusUnknownError: return "MIOPEN_STATUS_UNKNOWN_ERROR";
        default: return "MIOPEN_STATUS_UNKNOWN_ERROR";
        }
    */
}

