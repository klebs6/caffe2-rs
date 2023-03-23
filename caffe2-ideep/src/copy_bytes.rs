crate::ix!();

#[inline] pub fn copy_bytes_wrapper(
    nbytes:      usize,
    src:         *const c_void,
    src_device:  Device,
    dst:         *mut c_void,
    dst_device:  Device)  
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
