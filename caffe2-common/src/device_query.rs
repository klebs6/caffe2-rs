crate::ix!();

/**
  | Runs a device query function and prints
  | out the results to LOG(INFO).
  |
  */
#[inline] pub fn device_query(device: i32)  {
    
    todo!();
    /*
        const cudaDeviceProp& prop = GetDeviceProperty(device);
      std::stringstream ss;
      ss << std::endl;
      ss << "Device id:                     " << device << std::endl;
      ss << "Major revision number:         " << prop.major << std::endl;
      ss << "Minor revision number:         " << prop.minor << std::endl;
      ss << "Name:                          " << prop.name << std::endl;
      ss << "Total global memory:           " << prop.totalGlobalMem << std::endl;
      ss << "Total shared memory per block: " << prop.sharedMemPerBlock
         << std::endl;
      ss << "Total registers per block:     " << prop.regsPerBlock << std::endl;
      ss << "Warp size:                     " << prop.warpSize << std::endl;
    #ifndef __HIP_PLATFORM_HCC__
      ss << "Maximum memory pitch:          " << prop.memPitch << std::endl;
    #endif
      ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock
         << std::endl;
      ss << "Maximum dimension of block:    "
         << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
         << prop.maxThreadsDim[2] << std::endl;
      ss << "Maximum dimension of grid:     "
         << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
         << prop.maxGridSize[2] << std::endl;
      ss << "Clock rate:                    " << prop.clockRate << std::endl;
      ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
    #ifndef __HIP_PLATFORM_HCC__
      ss << "Texture alignment:             " << prop.textureAlignment << std::endl;
      ss << "Concurrent copy and execution: "
         << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
    #endif
      ss << "Number of multiprocessors:     " << prop.multiProcessorCount
         << std::endl;
    #ifndef __HIP_PLATFORM_HCC__
      ss << "Kernel execution timeout:      "
         << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    #endif
      LOG(INFO) << ss.str();
      return;
    */
}
