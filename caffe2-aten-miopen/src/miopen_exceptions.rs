crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Exceptions.h]

pub struct MiOpenException {
    base:   RuntimeError,
    status: MiOpenStatus,
}

impl MiOpenException {
    
    pub fn new(
        status: MiOpenStatus,
        msg:    *const u8) -> Self {
    
        todo!();
        /*
        : runtime_error(msg),
        : status(status),

        
        */
    }
    
    pub fn new(
        status: MiOpenStatus,
        msg:    &String) -> Self {
    
        todo!();
        /*
        : runtime_error(msg),
        : status(status),

        
        */
    }
}

#[inline] pub fn MIOPEN_CHECK(status: MiOpenStatus)  {
    
    todo!();
        /*
            if (status != miopenStatusSuccess) {
        if (status == miopenStatusNotImplemented) {
            throw miopen_exception(status, string(miopenGetErrorString(status)) +
                    ". This error may appear if you passed in a non-contiguous input.");
        }
        throw miopen_exception(status, miopenGetErrorString(status));
      }
        */
}

#[inline] pub fn HIP_CHECK(error: HipError)  {
    
    todo!();
        /*
            if (error != hipSuccess) {
        string msg("HIP error: ");
        msg += hipGetErrorString(error);
        throw runtime_error(msg);
      }
        */
}
