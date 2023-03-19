crate::ix!();

#[inline] pub fn copy_bytes_impl(
    nbytes: usize,
    src:    *const c_void,
    dst:    *mut c_void)  
{
    todo!();
    /*
        if (nbytes == 0) {
        return;
      }
      CAFFE_ENFORCE(src);
      CAFFE_ENFORCE(dst);
      memcpy(dst, src, nbytes);
    */
}

#[inline] pub fn copy_bytes_wrapper(
    nbytes:     usize,
    src:        *const c_void,
    src_device: Device,
    dst:        *mut c_void,
    dst_device: Device)  
{
    todo!();
    /*
        CopyBytesImpl(nbytes, src, dst);
    */
}

register_context!{
    DeviceType::CPU, 
    CPUContext
}

register_copy_bytes_function!{
    /*
    DeviceType::CPU,
    DeviceType::CPU,
    caffe2::CopyBytesWrapper
    */
}
