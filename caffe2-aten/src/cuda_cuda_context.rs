/*!
  | A common CUDA interface for ATen.
  | 
  | This interface is distinct from CUDAHooks,
  | which defines an interface that links
  | to both CPU-only and CUDA builds. That
  | interface is intended for runtime dispatch
  | and should be used from files that are
  | included in both CPU-only and
  | 
  | CUDA builds.
  | 
  | CUDAContext, on the other hand, should
  | be preferred by files only included
  | in
  | 
  | CUDA builds. It is intended to expose
  | CUDA functionality in a consistent
  | manner.
  | 
  | This means there is some overlap between
  | the CUDAContext and CUDAHooks, but
  | the choice of which to use is simple:
  | use CUDAContext when in a CUDA-only
  | file, use CUDAHooks otherwise.
  | 
  | -----------
  | @note
  | 
  | CUDAContext simply defines an interface
  | with no associated class.
  | 
  | It is expected that the modules whose
  | functions compose this interface will
  | manage their own state. There is only
  | a single CUDA context/state.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDAContext.h]

/**
  | DEPRECATED: use device_count() instead
  |
  */
#[inline] pub fn get_num_gpu_s() -> i64 {
    
    todo!();
        /*
            return c10::cuda::device_count();
        */
}

/**
  | CUDA is available if we compiled with
  | CUDA, and there are one or more devices.
  | If we compiled with CUDA but there is
  | a driver problem, etc., this function
  | will report CUDA is not available (rather
  | than raise an error.)
  |
  */
#[inline] pub fn is_available() -> bool {
    
    todo!();
        /*
            return c10::cuda::device_count() > 0;
        */
}

/* --------------------- Handles  --------------------- */

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDAContext.cpp]

lazy_static!{
    /*
    DeviceIndex num_gpus = -1;
    std::once_flag init_flag;
    std::deque<std::once_flag> device_flags;
    std::vector<cudaDeviceProp> device_properties;
    */
}

pub fn init_cuda_context_vectors()  {
    
    todo!();
        /*
            num_gpus = c10::cuda::device_count();
      device_flags.resize(num_gpus);
      device_properties.resize(num_gpus);
        */
}

pub fn init_device_property(device_index: DeviceIndex)  {
    
    todo!();
        /*
            cudaDeviceProp device_prop;
      AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
      device_properties[device_index] = device_prop;
        */
}

/**
  | We need this function to force the linking
  | against torch_cuda(_cpp) on Windows.
  |
  | If you need to modify this function, please
  | specify a new function and apply the changes
  | according to
  | https://github.com/pytorch/pytorch/pull/34288.
  |
  | Related issue:
  | https://github.com/pytorch/pytorch/issues/31611.
  |
  | Device info 
  */
pub fn warp_size() -> i32 {
    
    todo!();
        /*
            return getCurrentDeviceProperties()->warpSize;
        */
}

pub fn get_current_device_properties() -> *mut CudaDeviceProp {
    
    todo!();
        /*
            auto device = c10::cuda::current_device();
      return getDeviceProperties(device);
        */
}

pub fn get_device_properties(device: i64) -> *mut CudaDeviceProp {
    
    todo!();
        /*
            std::call_once(init_flag, initCUDAContextVectors);
      if (device == -1) device = c10::cuda::current_device();
      AT_ASSERT(device >= 0 && device < num_gpus);
      std::call_once(device_flags[device], initDeviceProperty, device);
      return &device_properties[device];
        */
}

pub fn can_device_access_peer(
        device:      i64,
        peer_device: i64) -> bool {
    
    todo!();
        /*
            std::call_once(init_flag, initCUDAContextVectors);
      if (device == -1) device = c10::cuda::current_device();
      AT_ASSERT(device >= 0 && device < num_gpus);
      AT_ASSERT(peer_device >= 0 && peer_device < num_gpus);
      int can_access = 0;
      AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
      return can_access != 0;
        */
}

pub fn get_cuda_device_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return c10::cuda::CUDACachingAllocator::get();
        */
}
