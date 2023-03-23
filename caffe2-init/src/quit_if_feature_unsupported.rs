crate::ix!();

#[inline] pub fn quit_if_feature_unsupported(
    cpu_has_feature: bool,
    feature:         &str)  
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
