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

