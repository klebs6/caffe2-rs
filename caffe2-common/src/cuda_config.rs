crate::ix!();

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

pub const kCUDAGridDimMaxX: i32 = 2147483647;
pub const kCUDAGridDimMaxY: i32 = 65535;
pub const kCUDAGridDimMaxZ: i32 = 65535;

