crate::ix!();

pub const gDefaultGPUID: i32 = 0;

#[inline] pub fn set_defaultGPUID(deviceid: i32)  {
    
    todo!();
    /*
        CAFFE_ENFORCE_LT(
          deviceid,
          NumCudaDevices(),
          "The default gpu id should be smaller than the number of gpus "
          "on this machine: ",
          deviceid,
          " vs ",
          NumCudaDevices());
      gDefaultGPUID = deviceid;
    */
}

#[inline] pub fn get_defaultGPUID() -> i32 {
    
    todo!();
    /*
        return gDefaultGPUID;
    */
}
