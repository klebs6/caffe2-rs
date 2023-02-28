crate::ix!();

/**
  | cuda major revision number below which
  | fp16 compute is not supoorted
  |
  */
#[cfg(__hip_platform_hcc__)]
pub const kFp16CUDADevicePropMajor: i32 = 6;

#[cfg(not(__hip_platform_hcc__))]
pub const kFp16CUDADevicePropMajor: i32 = 3;

/**
  | The maximum number of peers that each
  | gpu can have when doing p2p setup.
  | 
  | Currently, according to NVidia documentation,
  | each device can support a system-wide
  | maximum of eight peer connections.
  | 
  | When Caffe2 sets up peer access resources,
  | if we have more than 8 gpus, we will enable
  | peer access in groups of 8.
  |
  */
pub const CAFFE2_CUDA_MAX_PEER_SIZE: usize = 8;

#[cfg(cuda_version_gte_10000)]
pub type CAFFE2_CUDA_PTRATTR_MEMTYPE = type_;

#[cfg(not(cuda_version_gte_10000))]
pub type CAFFE2_CUDA_PTRATTR_MEMTYPE = MemoryType;

/**
  | A runtime function to report the cuda
  | version that Caffe2 is built with.
  |
  */
#[inline] pub fn cuda_version() -> i32 {
    
    todo!();
    /*
        return CUDA_VERSION;
    */
}

/**
  | Check if the current running session
  | has a cuda gpu present.
  | 
  | -----------
  | @note
  | 
  | this is different from having caffe2
  | built with cuda.
  | 
  | Building Caffe2 with cuda only guarantees
  | that this function exists.
  | 
  | If there are no cuda gpus present in the
  | machine, or there are hardware configuration
  | problems like an insufficient driver,
  | this function will still return false,
  | meaning that there is no usable GPU present.
  | 
  | In the open source build, it is possible
  | that
  | 
  | Caffe2's GPU code is dynamically loaded,
  | and as a result a library could be only
  | linked to the
  | 
  | CPU code, but want to test if cuda is later
  | available or not.
  | 
  | In this case, one should use HasCudaRuntime()
  | from common.h.
  |
  */
#[inline] pub fn has_cuda_gpu() -> bool {
    
    todo!();
    /*
        return NumCudaDevices() > 0;
    */
}

/// CUDA: various checks for different function calls.
#[macro_export] macro_rules! cuda_enforce {
    ($condition:ident, $($arg:ident),*) => {
        /*
        cudaError_t error = condition;   
        CAFFE_ENFORCE_EQ(                
            error,                       
            cudaSuccess,                 
            "Error at: ",                
            __FILE__,                    
            ":",                         
            __LINE__,                    
            ": ",                        
            cudaGetErrorString(error),   
            ##__VA_ARGS__);              
        */
    }
 }

#[macro_export] macro_rules! cuda_check {
    ($condition:ident) => {
        /*
        cudaError_t error = condition;                            
        CHECK(error == cudaSuccess) << cudaGetErrorString(error); 
        */
    }
 }

#[macro_export] macro_rules! cuda_driverapi_enforce {
    ($condition:ident) => {
        /*
        CUresult result = condition;                                     
        if (result != CUDA_SUCCESS) {                                    
            const char* msg;                                               
            cuGetErrorName(result, &msg);                                  
            CAFFE_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg); 
        }                                                                
        */
    }
}

#[macro_export] macro_rules! cuda_driverapi_check {
    ($condition:ident) => {
        /*
        CUresult result = condition;                                        
        if (result != CUDA_SUCCESS) {                                       
            const char* msg;                                                  
            cuGetErrorName(result, &msg);                                     
            LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": " 
                << msg;                                                
        }                                                                   
        */
    }
}

#[macro_export] macro_rules! cublas_enforce {
    ($condition:ident) => {
        todo!();
        /*
        cublasStatus_t status = condition;           
        CAFFE_ENFORCE_EQ(                            
            status,                                  
            CUBLAS_STATUS_SUCCESS,                   
            "Error at: ",                            
            __FILE__,                                
            ":",                                     
            __LINE__,                                
            ": ",                                    
            ::caffe2::cublasGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! cublas_check {
    ($condition:ident) => {
        todo!();
        /*
        cublasStatus_t status = condition;             
        CHECK(status == CUBLAS_STATUS_SUCCESS)         
            << ::caffe2::cublasGetErrorString(status); 
        */
    }
}

#[macro_export] macro_rules! curand_enforce {
    ($condition:ident) => {
        todo!();
        /*
        curandStatus_t status = condition;           
        CAFFE_ENFORCE_EQ(                            
            status,                                  
            CURAND_STATUS_SUCCESS,                   
            "Error at: ",                            
            __FILE__,                                
            ":",                                     
            __LINE__,                                
            ": ",                                    
            ::caffe2::curandGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! curand_check {
    ($condition:ident) => {
        todo!();
        /*
        curandStatus_t status = condition;             
        CHECK(status == CURAND_STATUS_SUCCESS)         
            << ::caffe2::curandGetErrorString(status); 
        */
    }
}

#[macro_export] macro_rules! cuda_1d_kernel_loop {
    ($i:ident, $n:ident) => {
        todo!();
        /*
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); 
            i += blockDim.x * gridDim.x)
        */
    }
}

#[macro_export] macro_rules! cuda_2d_kernel_loop {
    ($i:ident, $n:ident, $j:ident, $m:ident) => {
        todo!();
        /*
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)
        */
    }
}

/*
  | The following helper functions are
  | here so that you can write a kernel call
  | when you are not particularly interested
  | in maxing out the kernels' performance.
  | Usually, this will give you a reasonable
  | speed, but if you really want to find
  | the best performance, it is advised
  | that you tune the size of the blocks and
  | grids more reasonably.
  | 
  | A legacy note: this is derived from the
  | old good
  | 
  | Caffe days, when I simply hard-coded
  | the number of threads and wanted to keep
  | backward compatibility for different
  | computation capabilities.
  | 
  | For more info on CUDA compute capabilities,
  | visit the NVidia website at:
  | 
  | http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
  |
  */

/**
  | The number of cuda threads to use. Since
  | work is assigned to SMs at the granularity
  | of a block, 128 is chosen to allow utilizing
  | more SMs for smaller input sizes. 1D
  | grid
  |
  */
pub const CAFFE_CUDA_NUM_THREADS: i32 = 128;

/// 2D grid
pub const CAFFE_CUDA_NUM_THREADS_2D_DIMX: i32 = 16;
pub const CAFFE_CUDA_NUM_THREADS_2D_DIMY: i32 = 16;

/**
  | The maximum number of blocks to use in
  | the default kernel call. We set it to
  | 4096 which would work for compute capability
  | 2.x (where 65536 is the limit).
  | 
  | This number is very carelessly chosen.
  | Ideally, one would like to look at the
  | hardware at runtime, and pick the number
  | of blocks that makes most sense for the
  | specific runtime environment. This
  | is a todo item. 1D grid
  |
  */
pub const CAFFE_MAXIMUM_NUM_BLOCKS: i32 = 4096;

/// 2D grid
pub const CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX: i32 = 128;
pub const CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY: i32 = 128;

pub const kCUDAGridDimMaxX: i32 = 2147483647;
pub const kCUDAGridDimMaxY: i32 = 65535;
pub const kCUDAGridDimMaxZ: i32 = 65535;

/**
  | @brief
  | 
  | Compute the number of blocks needed
  | to run N threads.
  |
  */
#[inline] pub fn caffe_get_blocks(n: i32) -> i32 {
    
    todo!();
    /*
        return std::max(
          std::min(
              (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
              CAFFE_MAXIMUM_NUM_BLOCKS),
          // Use at least 1 block, since CUDA does not allow empty block
          1);
    */
}

/**
  | @brief
  | 
  | Compute the number of blocks needed
  | to run N threads for a 2D grid
  |
  */
#[inline] pub fn caffe_get_blocks_2d(n: i32, m: i32) -> Dim3 {
    
    todo!();
    /*
        dim3 grid;
      // Not calling the 1D version for each dim to keep all constants as literals

      grid.x = std::max(
          std::min(
              (N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) /
                  CAFFE_CUDA_NUM_THREADS_2D_DIMX,
              CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
          // Use at least 1 block, since CUDA does not allow empty block
          1);

      grid.y = std::max(
          std::min(
              (N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) /
                  CAFFE_CUDA_NUM_THREADS_2D_DIMY,
              CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
          // Use at least 1 block, since CUDA does not allow empty block
          1);

      return grid;
    */
}

pub struct SimpleArray<T, const N: usize> {
    data: [T; N],
}

pub const kCUDATensorMaxDims: usize = 8;

#[macro_export] macro_rules! dispatch_function_by_value_with_type_1 {
    ($val:ident, $Func:ident, $T:ident, $($arg:ident),*) => {
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                    
        switch (val) {                                                
            case 1: {                                                   
                Func<T, 1>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 2: {                                                   
                Func<T, 2>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 3: {                                                   
                Func<T, 3>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 4: {                                                   
                Func<T, 4>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 5: {                                                   
                Func<T, 5>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 6: {                                                   
                Func<T, 6>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 7: {                                                   
                Func<T, 7>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 8: {                                                   
                Func<T, 8>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            default: {                                                  
                break;                                                    
            }                                                           
        }                                                             
        */
    }
}

#[macro_export] macro_rules! dispatch_function_by_value_with_type_2 {
    ($val:ident, 
    $Func:ident, 
    $T1:ident, 
    $T2:ident, 
    $($arg:ident),*) => { 
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                         
        switch (val) {                                                     
            case 1: {                                                        
                Func<T1, T2, 1>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 2: {                                                        
                Func<T1, T2, 2>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 3: {                                                        
                Func<T1, T2, 3>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 4: {                                                        
                Func<T1, T2, 4>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 5: {                                                        
                Func<T1, T2, 5>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 6: {                                                        
                Func<T1, T2, 6>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 7: {                                                        
                Func<T1, T2, 7>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 8: {                                                        
                Func<T1, T2, 8>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            default: {                                                       
                break;                                                         
            }                                                                
        }                                                                  
        */
    }
}

#[macro_export] macro_rules! dispatch_function_by_value_with_type_3 {
    ($val:ident, $Func:ident, $T1:ident, $T2:ident, $T3:ident, $($arg:ident),*) => {
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                             
        switch (val) {                                                         
            case 1: {                                                            
                Func<T1, T2, T3, 1>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 2: {                                                            
                Func<T1, T2, T3, 2>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 3: {                                                            
                Func<T1, T2, T3, 3>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 4: {                                                            
                Func<T1, T2, T3, 4>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 5: {                                                            
                Func<T1, T2, T3, 5>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 6: {                                                            
                Func<T1, T2, T3, 6>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 7: {                                                            
                Func<T1, T2, T3, 7>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 8: {                                                            
                Func<T1, T2, T3, 8>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            default: {                                                           
                break;                                                             
            }                                                                    
        }                                                                      
        */
    }
}

/**
  | Returns the number of devices.
  |
  */
#[inline] pub fn num_cuda_devices() -> i32 {
    
    todo!();
    /*
        if (getenv("CAFFE2_DEBUG_CUDA_INIT_ORDER")) {
        static bool first = true;
        if (first) {
          first = false;
          std::cerr << "DEBUG: caffe2::NumCudaDevices() invoked for the first time"
                    << std::endl;
        }
      }
      // It logs warnings on first run
      return CudaDeviceCount();
    */
}

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

/**
  | Gets the current GPU id. This is a simple
  | wrapper around cudaGetDevice().
  |
  */
#[inline] pub fn caffe_cuda_get_device() -> i32 {
    
    todo!();
    /*
        int gpu_id = 0;
      CUDA_ENFORCE(cudaGetDevice(&gpu_id));
      return gpu_id;
    */
}

/**
  | Gets the current GPU id. This is a simple
  | wrapper around cudaGetDevice().
  |
  */
#[inline] pub fn caffe_cuda_set_device(id: i32)  {
    
    todo!();
    /*
        CUDA_ENFORCE(cudaSetDevice(id));
    */
}

/**
  | Gets the GPU id that the current pointer
  | is located at.
  |
  */
#[inline] pub fn get_gpuid_for_pointer(ptr: *const c_void) -> i32 {
    
    todo!();
    /*
        cudaPointerAttributes attr;
      cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

      if (err == cudaErrorInvalidValue) {
        // Occurs when the pointer is in the CPU address space that is
        // unmanaged by CUDA; make sure the last error state is cleared,
        // since it is persistent
        err = cudaGetLastError();
        CHECK(err == cudaErrorInvalidValue);
        return -1;
      }

      // Otherwise, there must be no error
      CUDA_ENFORCE(err);

      if (attr.CAFFE2_CUDA_PTRATTR_MEMTYPE == cudaMemoryTypeHost) {
        return -1;
      }

      return attr.device;
    */
}

pub struct CudaDevicePropWrapper {
    props: Vec<CudaDeviceProp>,
}

impl Default for CudaDevicePropWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            : props(NumCudaDevices()) 
                  for (int i = 0; i < NumCudaDevices(); ++i) {
                      CUDA_ENFORCE(cudaGetDeviceProperties(&props[i], i));
                  
        */
    }
}

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

/**
  | Return a peer access pattern by returning
  | a matrix (in the format of a nested vector)
  | of boolean values specifying whether
  | peer access is possible.
  | 
  | This function returns false if anything
  | wrong happens during the query of the
  | GPU access pattern.
  |
  */
#[inline] pub fn get_cuda_peer_access_pattern(pattern: *mut Vec<Vec<bool>>) -> bool {
    
    todo!();
    /*
        int gpu_count;
      if (cudaGetDeviceCount(&gpu_count) != cudaSuccess) return false;
      pattern->clear();
      pattern->resize(gpu_count, vector<bool>(gpu_count, false));
      for (int i = 0; i < gpu_count; ++i) {
        for (int j = 0; j < gpu_count; ++j) {
          int can_access = true;
          if (i != j) {
            if (cudaDeviceCanAccessPeer(&can_access, i, j)
                     != cudaSuccess) {
              return false;
            }
          }
          (*pattern)[i][j] = static_cast<bool>(can_access);
        }
      }
      return true;
    */
}

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

/**
  | Return a human readable cublas error
  | string.
  |
  */
#[inline] pub fn cublas_get_error_string(error: CuBlasStatus) -> *const u8 {
    
    todo!();
    /*
        switch (error) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    #ifndef __HIP_PLATFORM_HCC__
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    #else
      case rocblas_status_invalid_size:
        return "rocblas_status_invalid_size";
      case rocblas_status_perf_degraded:
        return "rocblas_status_perf_degraded";
      case rocblas_status_size_query_mismatch:
        return "rocblas_status_size_query_mismatch";
      case rocblas_status_size_increased:
        return "rocblas_status_size_increased";
      case rocblas_status_size_unchanged:
        return "rocblas_status_size_unchanged";
    #endif
      }
      // To suppress compiler warning.
      return "Unrecognized cublas error string";
    */
}

/**
  | Return a human readable curand error
  | string.
  |
  */
#[inline] pub fn curand_get_error_string(error: CuRandStatus) -> *const u8 {
    
    todo!();
    /*
        switch (error) {
      case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
      case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
      case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
      case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
      case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    #ifdef __HIP_PLATFORM_HCC__
      case HIPRAND_STATUS_NOT_IMPLEMENTED:
        return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    #endif
      }
      // To suppress compiler warning.
      return "Unrecognized curand error string";
    */
}

/**
  | Turn on the flag g_caffe2_has_cuda_linked
  | to true for HasCudaRuntime() function.
  |
  */
pub struct CudaRuntimeFlagFlipper { }

impl Default for CudaRuntimeFlagFlipper {
    
    fn default() -> Self {
        todo!();
        /*
            internal::SetCudaRuntimeFlag()
        */
    }
}

lazy_static!{
    static ref g_flipper: CudaRuntimeFlagFlipper = CudaRuntimeFlagFlipper::default();
}
