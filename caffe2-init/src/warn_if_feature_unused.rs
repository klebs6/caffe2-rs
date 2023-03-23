crate::ix!();

#[inline] pub fn warn_if_feature_unused(
    cpu_has_feature: bool,
    feature:         &str)  
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
