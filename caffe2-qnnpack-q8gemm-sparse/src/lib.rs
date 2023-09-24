#[macro_use] mod imports; use imports::*;

x!{q8gemm_sparse}
x!{q8gemm_sparse_8x4c1x4_packed_sse2}
x!{q8gemm_sparse_8x4_packa_aarch32_neon}
x!{q8gemm_sparse_4x4_packa_aarch32_neon}
x!{q8gemm_sparse_8x4_packa_aarch64_neon}
x!{q8gemm_sparse_8x4_packa_sse2}
x!{q8gemm_sparse_4x8c8x1_dq_packeda_aarch32_neon}
x!{q8gemm_sparse_8x8c1x4_dq_packeda_aarch64_neon}
x!{q8gemm_sparse_8x8c8x1_dq_packeda_aarch64_neon}
x!{q8gemm_sparse_4x8c1x4_dq_packeda_aarch32_neon}
x!{q8gemm_sparse_8x4c1x4_dq_packeda_sse2}
