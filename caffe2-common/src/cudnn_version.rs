crate::ix!();

#[inline] pub fn print_cudnn_info() -> bool {
    
    todo!();
    /*
        VLOG(1) << "Caffe2 is built with Cudnn version " << CUDNN_VERSION;
      return true;
    */
}

register_caffe2_init_function!{
   print_cudnn_info, 
   print_cudnn_info, 
   "Print Cudnn Info."
}

/**
  | TODO
  | 
  | cudnn_sys::cudnnGetVersion();
  |
  */
pub const CUDNN_VERSION: usize = 5000; 

/**
  |Caffe2 requires cudnn version 5.0 or above.
  |
  |CUDNN version under 6.0 is supported at best
  |effort.
  |
  |We strongly encourage you to move to 6.0 and
  |above.
  |
  |This message is intended to annoy you enough to
  |update.
  */
const_assert!{ CUDNN_VERSION >= 5000 } 

#[macro_export] macro_rules! cudnn_version_min {
    ($major:ident, 
     $minor:ident, 
     $patch:ident) => {
        todo!();
        /*
        CUDNN_VERSION >= ((major) * 1000 + (minor) * 100 + (patch))
        */
    }
}

/// report the version of cuDNN Caffe2 was compiled with
///
#[inline] pub fn cudnn_compiled_version() -> usize {
    
    todo!();
    /*
        return CUDNN_VERSION;
    */
}

/**
  | report the runtime version of cuDNN
  |
  */
#[inline] pub fn cudnn_runtime_version() -> usize {
    
    todo!();
    /*
        return cudnnGetVersion();
    */
}

/**
  | Check compatibility of compiled and
  | runtime cuDNN versions
  |
  */
#[inline] pub fn check_cudnn_versions()  {
    
    todo!();
    /*
        // Version format is major*1000 + minor*100 + patch
      // If compiled with version < 7, major, minor and patch must all match
      // If compiled with version >= 7, then either
      //    runtime_version > compiled_version
      //    major and minor match
      bool version_match = cudnnCompiledVersion() == cudnnRuntimeVersion();
      bool compiled_with_7 = cudnnCompiledVersion() >= 7000;
      bool backwards_compatible_7 = compiled_with_7 && cudnnRuntimeVersion() >= cudnnCompiledVersion();
      bool patch_compatible = compiled_with_7 && (cudnnRuntimeVersion() / 100) == (cudnnCompiledVersion() / 100);
      CAFFE_ENFORCE(version_match || backwards_compatible_7 || patch_compatible,
                    "cuDNN compiled (", cudnnCompiledVersion(), ") and "
                    "runtime (", cudnnRuntimeVersion(), ") versions mismatch");
    */
}
