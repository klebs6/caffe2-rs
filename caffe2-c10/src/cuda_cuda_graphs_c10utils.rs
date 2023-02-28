crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAGraphsC10Utils.h]

/**
  | Cuda Graphs utils used by c10 and aten.
  | aten/cuda/CUDAGraphsUtils.cuh adds
  | utils used by aten only.
  |
  */
pub type CaptureId = u64;

/**
  | first is set if the instance is created by
  | CUDAGraph::capture_begin.
  |
  | second is set if the instance is created by
  | graph_pool_handle.
  |
  */
pub type MempoolId = (CaptureId,CaptureId);

/**
  | RAII guard for "cudaStreamCaptureMode",
  | a thread-local value that controls the
  | error-checking strictness of a capture.
  |
  */
#[cfg(CUDA_VERSION_GTE_11000)]
#[cfg(feature = "cuda")]
pub struct CudaStreamCaptureModeGuard {
    strictness: CudaStreamCaptureMode,
}

#[cfg(CUDA_VERSION_GTE_11000)]
impl Drop for CudaStreamCaptureModeGuard {

    fn drop(&mut self) {
        todo!();
        /*
            C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&strictness_));
        */
    }
}

#[cfg(CUDA_VERSION_GTE_11000)]
impl CudaStreamCaptureModeGuard {
    
    pub fn new(desired: CudaStreamCaptureMode) -> Self {
    
        todo!();
        /*


            strictness_ = desired;
        C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&strictness_));
        */
    }
}

/**
  | Protects against enum cudaStreamCaptureStatus
  | implementation changes.
  |
  | Some compilers seem not to like static_assert
  | without the messages.
  */

/// unexpected int(cudaStreamCaptureStatusNone) value"
#[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
const_assert!(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) == 0);

/// unexpected int(cudaStreamCaptureStatusActive) value"
#[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
const_assert!(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) == 1);

/// unexpected int(cudaStreamCaptureStatusInvalidated) value
#[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
const_assert!(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) == 2);

#[repr(i32)]
pub enum CaptureStatus {

    #[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
    None,//        = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone),

    #[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
    Active,//      = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive),

    #[cfg(all(CUDA_VERSION,CUDA_VERSION_GTE_11000))]
    Invalidated,// = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated)

    #[cfg(not(all(CUDA_VERSION,CUDA_VERSION_GTE_11000)))]
    None,//        = 0
}

impl fmt::Display for CaptureStatus {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            switch (status) {
        case CaptureStatus::None:
          os << "cudaStreamCaptureStatusNone";
          break;
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        case CaptureStatus::Active:
          os << "cudaStreamCaptureStatusActive";
          break;
        case CaptureStatus::Invalidated:
          os << "cudaStreamCaptureStatusInvalidated";
          break;
    #endif
        default:
          TORCH_INTERNAL_ASSERT(
              false, "Unknown Cuda graph CaptureStatus", int(status));
      }
      return os;
        */
    }
}

/// Use this version where you're sure a Cuda
/// context exists already.
///
#[inline] pub fn current_stream_capture_status_may_init_ctx() -> CaptureStatus {
    
    todo!();
        /*
            #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
      cudaStreamCaptureStatus is_capturing;
      C10_CUDA_CHECK(
          cudaStreamIsCapturing(getCurrentCudaStream(), &is_capturing));
      return CaptureStatus(is_capturing);
    #else
      return CaptureStatus::None;
    #endif
        */
}
