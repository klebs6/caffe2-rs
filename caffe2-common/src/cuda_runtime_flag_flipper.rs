crate::ix!();

/**
  | Turn on the flag g_caffe2_has_cuda_linked
  | to true for HasCudaRuntime() function.
  |
  */
pub struct CudaRuntimeFlagFlipper { }

impl Default for CudaRuntimeFlagFlipper {
    
    fn default() -> Self {
        todo!();
        /*
            internal::SetCudaRuntimeFlag()
        */
    }
}

lazy_static!{
    static ref g_flipper: CudaRuntimeFlagFlipper = CudaRuntimeFlagFlipper::default();
}
