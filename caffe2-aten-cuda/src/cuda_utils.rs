crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDAUtils.h]

/**
  | Check if every tensor in a list of tensors
  | matches the current device.
  |
  */
#[inline] pub fn check_device(ts: &[Tensor]) -> bool {
    
    todo!();
        /*
            if (ts.empty()) {
        return true;
      }
      Device curDevice = Device(kCUDA, current_device());
      for (const Tensor& t : ts) {
        if (t.device() != curDevice) return false;
      }
      return true;
        */
}
