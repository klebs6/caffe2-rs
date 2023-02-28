// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm_sparse/8x4c1x4-packed-sse2.h]

pub const MR:                  usize = 8;
pub const COL_BLOCK_SIZE:      usize = 4;
pub const PACKED_A_BLOCK_SIZE: usize = COL_BLOCK_SIZE * MR;
