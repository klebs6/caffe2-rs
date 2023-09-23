crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/mkl/Exceptions.h]

#[inline] pub fn MKL_DFTI_CHECK(status: MKL_INT)  {
    
    todo!();
        /*
            if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        ostringstream ss;
        ss << "MKL FFT error: " << DftiErrorMessage(status);
        throw runtime_error(ss.str());
      }
        */
}
