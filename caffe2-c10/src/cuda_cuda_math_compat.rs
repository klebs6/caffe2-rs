/*!
  | This file defines math functions compatible
  | across different gpu platforms (currently
  | Cuda and HIP).
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAMathCompat.h]

#[cfg(any(__CUDACC__,__HIPCC__))]
lazy_static!{
    /*
    #ifdef __HIPCC__
    #define __MATH_FUNCTIONS_DECL__ inline C10_DEVICE
    #else /* __HIPCC__ */
    #ifdef __CUDACC_RTC__
    #define __MATH_FUNCTIONS_DECL__ 
    #else /* __CUDACC_RTC__ */
    #define __MATH_FUNCTIONS_DECL__ static inline 
    #endif /* __CUDACC_RTC__ */
    #endif /* __HIPCC__ */
    */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn abs(x: f32) -> f32 {
    
    todo!();
        /*
            return ::fabsf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn abs(x: f64) -> f64 {
    
    todo!();
        /*
            return ::fabs(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn exp(x: f32) -> f32 {
    
    todo!();
        /*
            return ::expf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn exp(x: f64) -> f64 {
    
    todo!();
        /*
            return ::exp(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn ceil(x: f32) -> f32 {
    
    todo!();
        /*
            return ::ceilf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn ceil(x: f64) -> f64 {
    
    todo!();
        /*
            return ::ceil(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn copysign(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            #if defined(__CUDA_ARCH__) || defined(__HIPCC__)
      return ::copysignf(x, y);
    #else
      // copysign gets ICE/Segfaults with gcc 7.5/8 on arm64
      // (e.g. Jetson), see PyTorch PR #51834
      // This host function needs to be here for the compiler but is never used
      TORCH_INTERNAL_ASSERT(
          false, "CUDAMathCompat copysign should not run on the CPU");
    #endif
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn copysign(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            #if defined(__CUDA_ARCH__) || defined(__HIPCC__)
      return ::copysign(x, y);
    #else
      // see above
      TORCH_INTERNAL_ASSERT(
          false, "CUDAMathCompat copysign should not run on the CPU");
    #endif
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn floor(x: f32) -> f32 {
    
    todo!();
        /*
            return ::floorf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn floor(x: f64) -> f64 {
    
    todo!();
        /*
            return ::floor(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn log(x: f32) -> f32 {
    
    todo!();
        /*
            return ::logf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn log(x: f64) -> f64 {
    
    todo!();
        /*
            return ::log(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn log1p(x: f32) -> f32 {
    
    todo!();
        /*
            return ::log1pf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn log1p(x: f64) -> f64 {
    
    todo!();
        /*
            return ::log1p(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn max(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return ::fmaxf(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn max(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            return ::fmax(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn min(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return ::fminf(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn min(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            return ::fmin(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn pow(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return ::powf(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn pow(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            return ::pow(x, y);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn sincos(
        x:    f32,
        sptr: *mut f32,
        cptr: *mut f32)  {
    
    todo!();
        /*
            return ::sincosf(x, sptr, cptr);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn sincos(
        x:    f64,
        sptr: *mut f64,
        cptr: *mut f64)  {
    
    todo!();
        /*
            return ::sincos(x, sptr, cptr);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn sqrt(x: f32) -> f32 {
    
    todo!();
        /*
            return ::sqrtf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn sqrt(x: f64) -> f64 {
    
    todo!();
        /*
            return ::sqrt(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn rsqrt(x: f32) -> f32 {
    
    todo!();
        /*
            return ::rsqrtf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn rsqrt(x: f64) -> f64 {
    
    todo!();
        /*
            return ::rsqrt(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn tan(x: f32) -> f32 {
    
    todo!();
        /*
            return ::tanf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn tan(x: f64) -> f64 {
    
    todo!();
        /*
            return ::tan(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn tanh(x: f32) -> f32 {
    
    todo!();
        /*
            return ::tanhf(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn tanh(x: f64) -> f64 {
    
    todo!();
        /*
            return ::tanh(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn normcdf(x: f32) -> f32 {
    
    todo!();
        /*
            return ::normcdff(x);
        */
}

#[cfg(any(__CUDACC__,__HIPCC__))]
#[__MATH_FUNCTIONS_DECL__] 
pub fn normcdf(x: f64) -> f64 {
    
    todo!();
        /*
            return ::normcdf(x);
        */
}
