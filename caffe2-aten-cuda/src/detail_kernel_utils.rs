/*!
  | Contents of this file are copied from
  | 
  | THCUNN/common.h for the ease of porting
  | THCUNN functions into ATen.
  |
  */
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/KernelUtils.h]

/**
  | CUDA: grid stride looping
  |
  | i64 _i_n_d_e_x specifically prevents overflow in the loop increment.
  |
  | If input.numel() < INT_MAX, _i_n_d_e_x < INT_MAX, except after the final
  | iteration of the loop where _i_n_d_e_x += blockDim.x * gridDim.x can be
  | greater than INT_MAX.  But in that case _i_n_d_e_x >= n, so there are no
  | further iterations and the overflowed value in i=_i_n_d_e_x is not used.
  */
#[macro_export] macro_rules! cuda_kernel_loop_type {
    ($i:ident, $n:ident, $index_type:ident) => {
        /*
        
          i64 _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           
          for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)
        */
    }
}

#[macro_export] macro_rules! cuda_kernel_loop {
    ($i:ident, $n:ident) => {
        /*
                CUDA_KERNEL_LOOP_TYPE(i, n, int)
        */
    }
}

/// Use 1024 threads per block, which requires
/// cuda sm_2x or above
///
pub const CUDA_NUM_THREADS: i32 = 1024;

/// CUDA: number of blocks for threads.
///
#[inline] pub fn GET_BLOCKS(N: i64) -> i32 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
      constexpr i64 max_int = int::max;

      // Round up division for positive number that cannot cause integer overflow
      auto block_num = (N - 1) / CUDA_NUM_THREADS + 1;
      TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

      return static_cast<int>(block_num);
        */
}
