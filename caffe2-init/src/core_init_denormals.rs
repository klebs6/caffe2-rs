crate::ix!();

#[inline] pub fn caffe_2set_denormals(
    i: *mut i32,
    c: *mut *mut *mut u8) -> bool 
{
    
    todo!();
    /*
        if (FLAGS_caffe2_ftz) {
        VLOG(1) << "Setting FTZ";
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
      }
      if (FLAGS_caffe2_daz) {
        VLOG(1) << "Setting DAZ";
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
      }
      return true;
    */
}

register_caffe2_init_function!{
    caffe2_set_denormals,
    caffe2_set_denormals,
    "Set denormal settings."
}
