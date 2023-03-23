crate::ix!();

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
