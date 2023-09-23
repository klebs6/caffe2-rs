#[macro_use] mod imports; use imports::*;

x!{q8gemm_4x8_neon}
x!{q8gemm_sparse}
x!{q8gemm_sparse_8x4c1x4_packed_sse2}
x!{q8gemm_sparse_8x4_packa_aarch32_neon}
x!{q8gemm_4x8_aarch32_neon}
x!{q8gemm}
x!{q8gemm_4x8_dq_aarch32_neon}
x!{q8gemm_4x8c2_xzp_aarch32_neon}
x!{q8gemm_8x8_neon}
x!{q8gemm_6x4_neon}
x!{q8gemm_sparse_4x4_packa_aarch32_neon}
x!{q8gemm_sparse_8x4_packa_aarch64_neon}
x!{q8gemm_4x4c2_dq_sse2}
x!{q8gemm_sparse_8x4_packa_sse2}
x!{q8gemm_sparse_4x8c8x1_dq_packeda_aarch32_neon}
x!{q8gemm_sparse_8x8c1x4_dq_packeda_aarch64_neon}
x!{q8gemm_8x8_dq_aarch64_neon}
x!{q8gemm_4x8_dq_neon}
x!{q8gemm_8x8_aarch64_neon}
x!{q8gemm_sparse_8x8c8x1_dq_packeda_aarch64_neon}
x!{q8gemm_4x8c2_xzp_neon}
x!{q8gemm_4x4c2_sse2}
x!{q8gemm_2x4c8_sse2}
x!{q8gemm_sparse_4x8c1x4_dq_packeda_aarch32_neon}
x!{q8gemm_sparse_8x4c1x4_dq_packeda_sse2}
x!{q8gemm_4x_sumrows_neon}