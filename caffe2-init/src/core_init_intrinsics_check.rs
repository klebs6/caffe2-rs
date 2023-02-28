crate::ix!();

#[inline] pub fn quit_if_feature_unsupported(
    cpu_has_feature: bool,
    feature: &String)  
{
    todo!();
    /*
        VLOG(1) << "Caffe2 built with " << feature << ".";
      if (!cpu_has_feature) {
        string err_string =
            "The Caffe2 binary is compiled with CPU feature " + feature +
            ", but your CPU does not support it. This will lead to segfaults "
            "on your machine, such as SIGILL 'illegal instructions' on Linux. "
            "As a result Caffe2 will preemptively quit. Please install or "
            "build a Caffe2 binary with the feature turned off.";
        if (FLAGS_caffe2_quit_on_unsupported_cpu_feature) {
          LOG(FATAL) << err_string;
        } else {
          LOG(ERROR) << err_string;
        }
      }
    */
}

#[inline] pub fn warn_if_feature_unused(
    cpu_has_feature: bool,
    feature: &String)  
{
    todo!();
    /*
        VLOG(1) << "Caffe2 not built with " << feature << ".";
      if (cpu_has_feature) {
    #ifdef CAFFE2_NO_CROSS_ARCH_WARNING
        // When cross-compiling single binary for multiple archs - turns off the
        // annoying warning
        VLOG(1)
    #else
        LOG(ERROR)
    #endif
            << "CPU feature " << feature
            << " is present on your machine, "
               "but the Caffe2 binary is not compiled with it. It means you "
               "may not get the full speed of your CPU.";
      }
    */
}

#[inline] pub fn caffe_2check_intrinsics_features(i: *mut i32, c: *mut *mut *mut u8) -> bool {
    
    todo!();
    /*
        #ifdef __AVX__
      QuitIfFeatureUnsupported(GetCpuId().avx(), "avx");
    #else
      WarnIfFeatureUnused(GetCpuId().avx(), "avx");
    #endif

    #ifdef __AVX2__
      QuitIfFeatureUnsupported(GetCpuId().avx2(), "avx2");
    #else
      WarnIfFeatureUnused(GetCpuId().avx2(), "avx2");
    #endif

    #ifdef __FMA__
      QuitIfFeatureUnsupported(GetCpuId().fma(), "fma");
    #else
      WarnIfFeatureUnused(GetCpuId().fma(), "fma");
    #endif

      return true;
    */
}

register_caffe2_init_function!{
    caffe2_check_intrinsics_features,
    caffe2_check_intrinsics_features,
    "Check intrinsics compatibility between the CPU feature and the binary."
}
