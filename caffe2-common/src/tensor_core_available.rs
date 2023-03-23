crate::ix!();

/**
  | Return the availability of TensorCores
  | for math
  |
  */
#[inline] pub fn tensor_core_available() -> bool {
    
    todo!();
    /*
        int device = CaffeCudaGetDevice();
      auto& prop = GetDeviceProperty(device);

      return prop.major >= 7;
    */
}
