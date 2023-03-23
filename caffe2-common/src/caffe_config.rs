crate::ix!();

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

