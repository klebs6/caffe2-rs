crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDADevice.h]

#[inline] pub fn get_device_from_ptr(ptr: *mut void) -> Device {
    
    todo!();
        /*
            cudaPointerAttributes attr;
      AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
      return {DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
        */
}
