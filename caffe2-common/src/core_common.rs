crate::ix!();

pub type CaffeMap<Key,Value> = HashMap<Key,Value>;

/**
  | dynamic cast reroute: if RTTI is disabled,
  | go to reinterpret_cast
  |
  */
#[inline] pub fn dynamic_cast_if_rtti<Dst, Src>(ptr: Src) -> Dst {
    todo!();
    /*
        #ifdef __GXX_RTTI
      return dynamic_cast<Dst>(ptr);
    #else
      return static_cast<Dst>(ptr);
    #endif
    */
}

/**
  | SkipIndices are used in
  | operator_fallback_gpu.h and
  | operator_fallback_mkl.h as utility functions
  | that marks input / output indices to skip when
  | we use a CPU operator as the fallback of
  | GPU/MKL operator option.
  |
  | note: this is supposed to be a variadic
  | template
  */
pub trait SkipIndices<const V: i32> {

    fn contains_internal(&self, i: i32) -> bool {
        i == V
    }

    /*
      template <int First, int Second, int... Rest>
      static inline bool ContainsInternal(const int i) {
        return (i == First) || ContainsInternal<Second, Rest...>(i);
      }

      static inline bool Contains(const int i) {
        return ContainsInternal<values...>(i);
      }

      static inline bool Contains(const int /*i*/) {
        return false;
      }
    */
}

lazy_static!{

    /**
      | A global variable to mark if Caffe2 has
      | cuda linked to the current runtime.
      |
      | Do not directly use this variable, but
      | instead use the HasCudaRuntime() function
      | below.
      */
    static ref g_caffe2_has_cuda_linked: AtomicBool = AtomicBool::new(false);
    static ref g_caffe2_has_hip_linked:  AtomicBool = AtomicBool::new(false);

}

/**
  | HasCudaRuntime() tells the program whether the
  | binary has Cuda runtime linked.
  |
  | This function should not be used in static
  | initialization functions as the underlying
  | boolean variable is going to be switched on
  | when one loads libtorch_gpu.so.
  */
#[inline] pub fn has_cuda_runtime() -> bool {
    
    todo!();
    /*
        return g_caffe2_has_cuda_linked.load();
    */
}

#[inline] pub fn has_hip_runtime() -> bool {
    
    todo!();
    /*
        return g_caffe2_has_hip_linked.load();
    */
}

/**
  | Sets the Cuda Runtime flag that is used by
  | HasCudaRuntime().
  |
  | You should never use this function - it is
  | only used by the Caffe2 gpu code to notify
  | Caffe2 core that cuda runtime has been loaded.
  */
#[inline] pub fn set_cuda_runtime_flag()  {
    
    todo!();
    /*
        g_caffe2_has_cuda_linked.store(true);
    */
}

#[inline] pub fn set_hip_runtime_flag()  {
    
    todo!();
    /*
        g_caffe2_has_hip_linked.store(true);
    */
}

/**
  | Returns which setting Caffe2 was configured
  | and built with (exported from CMake)
  |
  */
pub fn get_build_options<'a>() -> &'a HashMap<String,String> {
    todo!();
    /*
      static const std::map<string, string> kMap = CAFFE2_BUILD_STRINGS;
      return kMap;
    */
}
