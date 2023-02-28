/*!
  | Note: Explicit implementation of copysign for
  | f16 and BFloat16 is needed to workaround
  | g++-7/8 crash on aarch64, but also makes
  | copysign faster for the half-precision types
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/copysign.h]

/**
  | Implement copysign for half precision floats
  | using bit ops
  |
  | Sign is the most significant bit for both half
  | and bfloat16 types
  |
  */
#[inline] pub fn copysign_f16(a: f16, b: f16) -> f16 {
    
    todo!();
        /*
            return f16((a.x & 0x7fff) | (b.x & 0x8000), f16::from_bits());
        */
}

#[inline] pub fn copysign_bf16(a: bf16, b: bf16) -> bf16 {
    
    todo!();
        /*
            return BFloat16(
          (a.x & 0x7fff) | (b.x & 0x8000), BFloat16::from_bits());
        */
}
