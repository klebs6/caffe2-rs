/*!
  | !!!! PLEASE READ !!!!
  |
  | Minimize (transitively) included headers from
  | _avx*.cc because some of the functions defined
  | in the headers compiled with platform dependent
  | compiler options can be reused by other
  | translation units generating illegal
  | instruction run-time error.
  */

crate::ix!();

/**
  | Common utilities for writing performance
  | kernels and easy dispatching of different
  | backends.
  |
  */

/**
 | The general workflow shall be as follows, say we
 | want to implement a functionality called void
 | foo(int a, float b).
 |
 | In foo.h, do:
 |    void foo(int a, float b);
 |
 | In foo_avx512.cc, do:
 |    void foo__avx512(int a, float b) {
 |      [actual avx512 implementation]
 |    }
 |
 | In foo_avx2.cc, do:
 |    void foo__avx2(int a, float b) {
 |      [actual avx2 implementation]
 |    }
 |
 | In foo_avx.cc, do:
 |    void foo__avx(int a, float b) {
 |      [actual avx implementation]
 |    }
 |
 | In foo.cc, do:
 |    // The base implementation should *always* be provided.
 |    void foo__base(int a, float b) {
 |      [base, possibly slow implementation]
 |    }
 |    decltype(foo__base) foo__avx512;
 |    decltype(foo__base) foo__avx2;
 |    decltype(foo__base) foo__avx;
 |    void foo(int a, float b) {
 |      // You should always order things by their preference, faster
 |      // implementations earlier in the function.
 |      AVX512_DO(foo, a, b);
 |      AVX2_DO(foo, a, b);
 |      AVX_DO(foo, a, b);
 |      BASE_DO(foo, a, b);
 |    }
 |
 */

/**
  | Details: this functionality basically covers
  | the cases for both build time and run time
  | architecture support.
  |
  | During build time:
  |
  |    The build system should provide flags
  |    CAFFE2_PERF_WITH_AVX512,
  |    CAFFE2_PERF_WITH_AVX2, and
  |    CAFFE2_PERF_WITH_AVX that corresponds to the
  |    __AVX512F__, __AVX512DQ__, __AVX512VL__,
  |    __AVX2__, and __AVX__ flags the compiler
  |    provides. Note that we do not use the
  |    compiler flags but rely on the build system
  |    flags, because the common files (like foo.cc
  |    above) will always be built without
  |    __AVX512F__, __AVX512DQ__, __AVX512VL__,
  |    __AVX2__ and __AVX__.
  |
  | During run time:
  |
  |    we use cpuinfo to identify cpu support and
  |    run the proper functions.
  |
  | DO macros: these should be used in your entry
  | function, similar to foo() above, that routes
  | implementations based on CPU capability.
  */
#[macro_export] macro_rules! base_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
                return funcname##__base(__VA_ARGS__);
        */
    }
}


#[cfg(caffe2_perf_with_avx512)]
#[macro_export] macro_rules! avx512_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                                                
            static const bool isDo = cpuinfo_initialize() &&               
                cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512dq() && 
                cpuinfo_has_x86_avx512vl();                                
            if (isDo) {                                                    
              return funcname##__avx512(__VA_ARGS__);                      
            }                                                              
          }
        */
    }
}

#[cfg(not(caffe2_perf_with_avx512))]
#[macro_export] macro_rules! avx512_do {
    () => {
        todo!();
        /*
                (funcname, ...)
        */
    }
}

#[cfg(caffe2_perf_with_avx2)]
#[macro_export] macro_rules! avx2_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                                                          
            static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx2(); 
            if (isDo) {                                                              
              return funcname##__avx2(__VA_ARGS__);                                  
            }                                                                        
          }
        */
    }
}

#[cfg(caffe2_perf_with_avx2)]
#[macro_export] macro_rules! avx2_fma_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                                                            
            static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx2() && 
                cpuinfo_has_x86_fma3();                                                
            if (isDo) {                                                                
              return funcname##__avx2_fma(__VA_ARGS__);                                
            }                                                                          
          }
        */
    }
}

#[cfg(not(caffe2_perf_with_avx2))]
#[macro_export] macro_rules! avx2_do {
    () => {
        todo!();
        /*
                (funcname, ...)
        */
    }
}

#[cfg(not(caffe2_perf_with_avx2))]
#[macro_export] macro_rules! avx2_fma_do {
    () => {
        todo!();
        /*
                (funcname, ...)
        */
    }
}

#[cfg(caffe2_perf_with_avx)]
#[macro_export] macro_rules! avx_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                            
            static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx(); 
            if (isDo) {                                
              return funcname##__avx(__VA_ARGS__);     
            }                                          
          }
        */
    }
}

#[cfg(caffe2_perf_with_avx)]
#[macro_export] macro_rules! avx_f16c_do {
    ($funcname:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                                                           
            static const bool isDo = cpuinfo_initialize() && cpuinfo_has_x86_avx() && 
                cpuinfo_has_x86_f16c();                                               
            if (isDo) {                                                               
              return funcname##__avx_f16c(__VA_ARGS__);                               
            }                                                                         
          }
        */
    }
}

#[cfg(not(caffe2_perf_with_avx))]
#[macro_export] macro_rules! avx_do {
    () => {
        todo!();
        /*
                (funcname, ...)
        */
    }
}

#[cfg(not(caffe2_perf_with_avx))]
#[macro_export] macro_rules! avx_f16c_do {
    () => {
        todo!();
        /*
                (funcname, ...)
        */
    }
}
