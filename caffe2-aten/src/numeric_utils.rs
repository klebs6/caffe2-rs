/*!
  | isnan isn't performant to use on integral
  | types; it will (uselessly) convert to floating
  | point and then do the test.
  |
  | This function is.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/NumericUtils.h]

#[inline] pub fn integer_isnan<T: Integer>(val: T) -> bool {
    
    todo!();
        /*
            return false;
        */
}

#[inline] pub fn float_isnan<T: Float>(val: T) -> bool {
    
    todo!();
        /*
            #if defined(__CUDACC__) || defined(__HIPCC__)
      return ::isnan(val);
    #else
      return isnan(val);
    #endif
        */
}

#[inline] pub fn complex_isnan<T: Complex>(val: T) -> bool {
    
    todo!();
        /*
            return isnan(val.real()) || isnan(val.imag());
        */
}

#[inline] pub fn half_isnan(val: Half) -> bool {
    
    todo!();
        /*
            return _isnan(static_cast<float>(val));
        */
}

#[inline] pub fn bf16_isnan(val: bf16) -> bool {
    
    todo!();
        /*
            return _isnan(static_cast<float>(val));
        */
}

#[inline] pub fn exp<T>(x: T) -> T {

    todo!();
        /*
            static_assert(!is_same<T, double>::value, "this template must be used with float or less precise type");
    #if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
      // use __expf fast approximation for peak bandwidth
      return __expf(x);
    #else
      return ::exp(x);
    #endif
        */
}

#[inline] pub fn exp_double(x: f64) -> f64 {
    
    todo!();
        /*
            return ::exp(x);
        */
}

#[inline] pub fn log<T>(x: T) -> T {

    todo!();
        /*
            static_assert(!is_same<T, double>::value, "this template must be used with float or less precise type");
    #if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
      // use __logf fast approximation for peak bandwidth
      return __logf(x);
    #else
      return ::log(x);
    #endif
        */
}

#[inline] pub fn log_double(x: f64) -> f64 {
    
    todo!();
        /*
            return ::log(x);
        */
}

#[inline] pub fn tan<T>(x: T) -> T {

    todo!();
        /*
            static_assert(!is_same<T, double>::value, "this template must be used with float or less precise type");
    #if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
      // use __tanf fast approximation for peak bandwidth
      return __tanf(x);
    #else
      return ::tan(x);
    #endif
        */
}

#[inline] pub fn tan_double(x: f64) -> f64 {
    
    todo!();
        /*
            return ::tan(x);
        */
}
