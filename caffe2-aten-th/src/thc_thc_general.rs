crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCGeneral.h.in]

/**
  | Needed for hipMAGMA to correctly identify
  | implementation
  |
  */
#[cfg(all(USE_MAGMA,__HIP_PLATFORM_HCC__))]
pub const HAVE_HIP: usize = 1;

#[cfg(not(THAssert))]
#[macro_export] macro_rules! th_assert {
    ($exp:ident) => {
        /*
        
          do {                                                                  
            if (!(exp)) {                                                       
              _THError(__FILE__, __LINE__, "assert(%s) failed", #exp);          
            }                                                                   
          } while(0)
        */
    }
}


pub struct THCCudaResourcesPerDevice {

    /**
      | usize of scratch space per each stream
      | on this device available
      |
      */
    scratch_space_per_stream: usize,
}

#[macro_export] macro_rules! thc_assert_same_gpu {
    ($expr:ident) => {
        /*
            if (!expr) THError("arguments are located on different GPUs")
        */
    }
}

#[macro_export] macro_rules! th_cuda_check {
    ($err:ident) => {
        /*
            __THCudaCheck(err, __FILE__, __LINE__)
        */
    }
}

#[macro_export] macro_rules! th_cuda_check_warn {
    ($err:ident) => {
        /*
            __THCudaCheckWarn(err, __FILE__, __LINE__)
        */
    }
}

#[macro_export] macro_rules! th_cublas_check {
    ($err:ident) => {
        /*
            __THCublasCheck(err,  __FILE__, __LINE__)
        */
    }
}

#[macro_export] macro_rules! th_cusparse_check {
    ($err:ident) => {
        /*
                __THCusparseCheck(err,  __FILE__, __LINE__)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCGeneral.hpp]

/**
  | Global state of THC.
  |
  */
pub struct THCState {

    /**
      | Set of all allocated resources.
      |
      */
    resources_per_device: *mut THCCudaResourcesPerDevice,

    /**
      | Captured number of devices upon startup;
      | convenience for bounds checking
      |
      */
    num_devices:          i32,

    /**
      | Allocator using cudaMallocHost.
      | 
      | NB: cudaHostAllocator MUST implement
      | maybeGlobalBoundDeleter, because
      | we have a few use-cases where we need
      | to do raw allocations with them (for
      | Thrust).
      | 
      | TODO: Make this statically obvious
      |
      */
    cuda_host_allocator:  *mut Allocator,

    /**
      | Table of enabled peer-to-peer access
      | between directed pairs of GPUs.
      | 
      | If i accessing allocs on j is enabled,
      | p2pAccess[i][j] is 1; 0 otherwise.
      |
      */
    p2p_access_enabled:   *mut *mut i32,
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCGeneral.cpp]

/**
  | usize of scratch space available in global
  | memory per each SM + stream
  |
  */
#[macro_export] macro_rules! min_global_scratch_space_per_sm_stream {
    () => {
        /*
                4 * sizeof(float)
        */
    }
}

/* 
 | Minimum amount of scratch space per
 | device. Total scratch memory per device is
 | either this amount, or the # of SMs * the space
 | per SM defined above, whichever is greater.
 |
 */
#[macro_export] macro_rules! min_global_scratch_space_per_device {
    () => {
        /*
                32768 * sizeof(float)
        */
    }
}

/**
  | Maximum number of P2P connections (if
  | there are more than 9 then P2P is enabled
  | in groups of 8).
  |
  */
pub const THC_CUDA_MAX_PEER_SIZE: usize = 8;

pub fn thc_state_free(state: *mut THCState)  {
    
    todo!();
        /*
            free(state);
        */
}

pub fn thc_state_alloc() -> *mut THCState {
    
    todo!();
        /*
            THCState* state = (THCState*) calloc(1, sizeof(THCState));
      return state;
        */
}

pub fn th_cuda_init(state: *mut THCState)  {
    
    todo!();
        /*
            if (!state->cudaHostAllocator) {
        state->cudaHostAllocator = getTHCCachingHostAllocator();
      }

      // We want to throw if there are no GPUs
      int numDevices = static_cast<int>(device_count_ensure_non_zero());
      state->numDevices = numDevices;

      CUDACachingAllocator::init(numDevices);

      int device = 0;
      THCudaCheck(cudaGetDevice(&device));

      state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
        calloc(numDevices, sizeof(THCCudaResourcesPerDevice));

      // p2pAccessEnabled records if p2p copies are allowed between pairs of
      // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
      // "-1" (unknown).
      // Currently the max number of gpus in P2P group is 8, so if there are more
      // we enable P2P in groups of 8
      state->p2pAccessEnabled = (int**) calloc(numDevices, sizeof(int*));
      for (int i = 0; i < numDevices; ++i) {
        state->p2pAccessEnabled[i] = (int*) calloc(numDevices, sizeof(int));
        for (int j = 0; j < numDevices; ++j)
          if (i == j)
            state->p2pAccessEnabled[i][j] = 1;
          else
            state->p2pAccessEnabled[i][j] = -1;
      }

      for (int i = 0; i < numDevices; ++i) {
        THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
        THCudaCheck(cudaSetDevice(i));

        /* The scratch space that we want to have available per each device is
           based on the number of SMs available per device. We guarantee a
           minimum of 128kb of space per device, but to future-proof against
           future architectures that may have huge #s of SMs, we guarantee that
           we have at least 16 bytes for each SM. */
        int numSM = getDeviceProperties(i)->multiProcessorCount;
        usize sizePerStream =
          MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
          MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
          numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
        res->scratchSpacePerStream = sizePerStream;
      }

      /* Restore to previous device */
      THCudaCheck(cudaSetDevice(device));
        */
}

pub fn th_cuda_shutdown(state: *mut THCState)  {
    
    todo!();
        /*
            int deviceCount = 0;
      int prevDev = -1;
      THCudaCheck(cudaGetDevice(&prevDev));
      THCudaCheck(cudaGetDeviceCount(&deviceCount));

      /* cleanup p2p access state */
      for (int dev = 0; dev < deviceCount; ++dev) {
        free(state->p2pAccessEnabled[dev]);
      }
      free(state->p2pAccessEnabled);

      free(state->resourcesPerDevice);
      CUDACachingAllocator::emptyCache();
      if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
        THCCachingHostAllocator_emptyCache();
      }

      THCudaCheck(cudaSetDevice(prevDev));
        */
}

/**
  | If device `dev` can access allocations
  | on device `devToAccess`, this will
  | return 1; otherwise, 0.
  |
  */
pub fn thc_state_get_peer_to_peer_access(
    state:         *mut THCState,
    dev:           i32,
    dev_to_access: i32) -> i32 {

    todo!();
        /*
            if (dev < 0 || dev >= state->numDevices) {
        THError("%d is not a device", dev);
      }
      if (devToAccess < 0 || devToAccess >= state->numDevices) {
        THError("%d is not a device", devToAccess);
      }
      if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
        int prevDev = 0;
        THCudaCheck(cudaGetDevice(&prevDev));
        THCudaCheck(cudaSetDevice(dev));

        int access = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&access, dev, devToAccess));
        if (access) {
          cudaError_t err = cudaDeviceEnablePeerAccess(devToAccess, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            // ignore and clear the error if access was already enabled
            cudaGetLastError();
          } else {
            THCudaCheck(err);
          }
          state->p2pAccessEnabled[dev][devToAccess] = 1;
        } else {
          state->p2pAccessEnabled[dev][devToAccess] = 0;
        }

        THCudaCheck(cudaSetDevice(prevDev));
      }
      return state->p2pAccessEnabled[dev][devToAccess];
        */
}

pub fn thc_state_get_cuda_host_allocator(state: *mut THCState) -> *mut Allocator {
    
    todo!();
        /*
            return state->cudaHostAllocator;
        */
}

pub fn thc_state_get_device_resource_ptr(
    state:  *mut THCState,
    device: i32) -> *mut THCCudaResourcesPerDevice {
    
    todo!();
        /*
            /* `device` is a CUDA index */
      if (device >= state->numDevices || device < 0)
      {
        THError("%d is not a device", device + 1 /* back to Torch index */);
      }

      return &(state->resourcesPerDevice[device]);
        */
}

/**
  | For the current device and stream, returns
  | the allocated scratch space
  |
  */
pub fn thc_state_get_current_device_scratch_space_size(state: *mut THCState) -> usize {
    
    todo!();
        /*
            int device = -1;
      THCudaCheck(cudaGetDevice(&device));
      THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
      return res->scratchSpacePerStream;
        */
}

pub fn th_cuda_check(
    err:  CudaError,
    file: *const u8,
    line: i32)  {
    
    todo!();
        /*
            if(err != cudaSuccess)
      {
        static int alreadyFailed = 0;
        if(!alreadyFailed) {
          fprintf(stderr, "THCudaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
          alreadyFailed = 1;
        }
        _THError(file, line, "cuda runtime error (%d) : %s", err,
                 cudaGetErrorString(err));
      }
        */
}

pub fn th_cuda_check_warn(
    err:  CudaError,
    file: *const u8,
    line: i32)  {

    todo!();
        /*
            if(err != cudaSuccess)
      {
        fprintf(stderr, "THCudaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
      }
        */
}

pub fn th_cublas_check(
    status: CuBlasStatus,
    file:   *const u8,
    line:   i32)  {
    
    todo!();
        /*
            if(status != CUBLAS_STATUS_SUCCESS)
      {
        const char* errmsg = NULL;

        switch(status)
        {
          case CUBLAS_STATUS_NOT_INITIALIZED:
            errmsg = "library not initialized";
            break;

          case CUBLAS_STATUS_ALLOC_FAILED:
            errmsg = "resource allocation failed";
            break;

          case CUBLAS_STATUS_INVALID_VALUE:
            errmsg = "an invalid numeric value was used as an argument";
            break;

          case CUBLAS_STATUS_ARCH_MISMATCH:
            errmsg = "an absent device architectural feature is required";
            break;

    #ifndef __HIP_PLATFORM_HCC__
          case CUBLAS_STATUS_MAPPING_ERROR:
            errmsg = "an access to GPU memory space failed";
            break;

          case CUBLAS_STATUS_EXECUTION_FAILED:
            errmsg = "the GPU program failed to execute";
            break;
    #endif

          case CUBLAS_STATUS_INTERNAL_ERROR:
            errmsg = "an internal operation failed";
            break;

          default:
            errmsg = "unknown error";
            break;
        }

        _THError(file, line, "cublas runtime error : %s", errmsg);
      }
        */
}

pub fn th_cusparse_check(
    status: CuSparseStatus,
    file:   *const u8,
    line:   i32)  {
    
    todo!();
        /*
            if(status != CUSPARSE_STATUS_SUCCESS)
      {
        const char* errmsg = NULL;

        switch(status)
        {
          case CUSPARSE_STATUS_NOT_INITIALIZED:
            errmsg = "library not initialized";
            break;

          case CUSPARSE_STATUS_ALLOC_FAILED:
            errmsg = "resource allocation failed";
            break;

          case CUSPARSE_STATUS_INVALID_VALUE:
            errmsg = "an invalid numeric value was used as an argument";
            break;

          case CUSPARSE_STATUS_ARCH_MISMATCH:
            errmsg = "an absent device architectural feature is required";
            break;

          case CUSPARSE_STATUS_MAPPING_ERROR:
            errmsg = "an access to GPU memory space failed";
            break;

          case CUSPARSE_STATUS_EXECUTION_FAILED:
            errmsg = "the GPU program failed to execute";
            break;

          case CUSPARSE_STATUS_INTERNAL_ERROR:
            errmsg = "an internal operation failed";
            break;

          case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            errmsg = "the matrix type is not supported by this function";
            break;

          default:
            errmsg = "unknown error";
            break;
        }

        _THError(file, line, "cusparse runtime error : %s", errmsg);
      }
        */
}

pub fn th_cuda_malloc(
    state: *mut THCState,
    size:  usize)  {

    todo!();
        /*
            return CUDACachingAllocator::raw_alloc(size);
        */
}

pub fn th_cuda_free(
    state: *mut THCState,
    ptr:   *mut c_void)  {
    
    todo!();
        /*
            CUDACachingAllocator::raw_delete(ptr);
        */
}

pub fn th_cuda_host_alloc(
    state: *mut THCState,
    size:  usize) -> DataPtr {

    todo!();
        /*
            THCudaCheck(cudaGetLastError());
      Allocator* allocator = state->cudaHostAllocator;
      return allocator->allocate(size);
        */
}

pub fn th_cuda_host_record(
    state: *mut THCState,
    ptr:   *mut c_void)  {
    
    todo!();
        /*
            if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
        THCCachingHostAllocator_recordEvent(ptr, getCurrentCUDAStream());
      }
        */
}
