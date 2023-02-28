/*!
  | Android NDK platform < 21 with libstdc++ has
  | spotty C++11 support.
  |
  | Various hacks in this header allow the rest of
  | the codebase to use standard APIs.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/math_compat.h]

// Define float versions the same way as more recent libstdc++
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn acosh(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_acoshf(x);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn asinh(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_asinhf(x);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn atanh(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_atanhf(x);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn copysign(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_copysignf(x, y);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn erf(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_erff(x);
        */
}




#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn erfc(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_erfcf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn expm1(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_expm1f(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn fmax(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_fmaxf(x, y);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn fmin(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_fminf(x, y);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn lgamma(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_lgammaf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn log1p(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_log1pf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn nearbyint(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_nearbyintf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn remainder(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_remainderf(x, y);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn round(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_roundf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn tgamma(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_tgammaf(x);
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn trunc(x: f32) -> f32 {
    
    todo!();
        /*
            return __builtin_truncf(x);
        */
}

// __builtin_nexttoward isn't doesn't work.  It appears to try to
// link against the global nexttoward function, which is not present
// prior to API 18.  Just bail for now.
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn nexttoward(x: f32, y: f64) -> f32 {
    
    todo!();
        /*
            throw runtime_error("nexttoward is not present on older Android");
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn nexttoward(x: f64, y: i64) -> f64 {
    
    todo!();
        /*
            throw runtime_error("nexttoward is not present on older Android");
        */
}

/**
  | TODO: this function needs to be implemented
  | and tested. Currently just throw an
  | error.
  |
  */
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn hypot(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            throw runtime_error("hypot is not implemented on older Android");
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn hypot(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            throw runtime_error("hypot is not implemented on older Android");
        */
}

/**
  | TODO: this function needs to be implemented
  | and tested. Currently just throw an
  | error.
  |
  */
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn igamma(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            throw runtime_error("igamma is not implemented on older Android");
        */
}


#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn igamma(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            throw runtime_error("igamma is not implemented on older Android");
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn igammac(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            throw runtime_error("igammac is not implemented on older Android");
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn igammac(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            throw runtime_error("igammac is not implemented on older Android");
        */
}

/**
  | TODO: this function needs to be implemented
  | and tested. Currently just throw an
  | error.
  |
  */
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn nextafter(x: f32, y: f32) -> f32 {
    
    todo!();
        /*
            throw runtime_error(
          "nextafter is not implemented on older Android");
        */
}



#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn nextafter(x: f64, y: f64) -> f64 {
    
    todo!();
        /*
            throw runtime_error(
          "nextafter is not implemented on older Android");
        */
}

/**
  | TODO: this function needs to be implemented
  | and tested. Currently just throw an
  | error.
  |
  */
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn exp2(x: f32) -> f32 {
    
    todo!();
        /*
            throw runtime_error("exp2 is not implemented on older Android");
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn exp2(x: f64) -> f64 {
    
    todo!();
        /*
            throw runtime_error("exp2 is not implemented on older Android");
        */
}

// Define integral versions the same way as more recent libstdc++
//
// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn acosh<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_acosh(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn asinh<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_asinh(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type 
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn atanh<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_atanh(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn erf<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_erf(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn erfc<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_erfc(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn expm1<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_expm1(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn lgamma<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_lgamma(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn log1p<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_log1p(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn nearbyint<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_nearbyint(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn round<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_round(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn tgamma<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_tgamma(x);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn trunc<T>(x: T) -> f64 {

    todo!();
        /*
            return __builtin_trunc(x);
        */
}

/**
  | Convoluted definition of these binary functions
  | for overloads other than (float,float) and
  | (double,double).
  |
  | Using a template from __gnu_cxx is dirty, but
  | this code is only enabled on a dead platform,
  | so there shouldn't be any risk of it breaking
  | due to updates.
  |
  */
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
lazy_static!{
    /*
    template <typename T, typename U>
    typename __gnu_cxx::__promote_2<T, U>::__type fmax(T x, U y) {
      typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
      return fmax(type(x), type(y));
    }
    template <typename T, typename U>
    typename __gnu_cxx::__promote_2<T, U>::__type fmin(T x, U y) {
      typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
      return fmin(type(x), type(y));
    }
    template <typename T, typename U>
    typename __gnu_cxx::__promote_2<T, U>::__type copysign(T x, U y) {
      typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
      return copysign(type(x), type(y));
    }
    template <typename T, typename U>
    typename __gnu_cxx::__promote_2<T, U>::__type remainder(T x, U y) {
      typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
      return remainder(type(x), type(y));
    }
    */
}

/// log2 is a macro on Android API < 21, so we
/// need to define it ourselves.
///
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn log2(arg: f32) -> f32 {
    
    todo!();
        /*
            return ::log(arg) / ::log(2.0);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn log2(arg: f64) -> f64 {
    
    todo!();
        /*
            return ::log(arg) / ::log(2.0);
        */
}

#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
#[inline] pub fn log2(arg: f64) -> f64 {
    
    todo!();
        /*
            return ::log(arg) / ::log(2.0);
        */
}

// typename enable_if<is_integral<T>::value, double>::type
#[cfg(all(__ANDROID__,__ANDROID_API__LT_21,__GLIBCXX__))]
pub fn log2<T>(x: T) -> f64 {

    todo!();
        /*
            return ::log(x) / ::log(2.0);
        */
}
