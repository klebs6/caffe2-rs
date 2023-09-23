crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/LaunchUtils.h]

/**
  | returns 2**floor(log2(n))
  |
  */
pub fn last_pow2(n: u32) -> i32 {
    
    todo!();
        /*
            n |= (n >> 1);
      n |= (n >> 2);
      n |= (n >> 4);
      n |= (n >> 8);
      n |= (n >> 16);
      return max<int>(1, n - (n >> 1));
        */
}
