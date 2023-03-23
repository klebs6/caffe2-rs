crate::ix!();

/**
  | Gets the device property for the given
  | device. This function is thread safe.
  | 
  | The initial run on this function is ~1ms/device;
  | however, the results are cached so subsequent
  | runs should be much faster.
  |
  */
#[inline] pub fn get_device_property<'a>(deviceid: i32) -> &'a CudaDeviceProp {
    
    todo!();
    /*
        // According to C++11 standard section 6.7, static local variable init is
      // thread safe. See
      //   https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11
      // for details.
      static CudaDevicePropWrapper props;
      CAFFE_ENFORCE_LT(
          deviceid,
          NumCudaDevices(),
          "The gpu id should be smaller than the number of gpus ",
          "on this machine: ",
          deviceid,
          " vs ",
          NumCudaDevices());
      return props.props[deviceid];
    */
}
