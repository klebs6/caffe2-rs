#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{perfkernels_adagrad_avx2}
x!{perfkernels_adagrad}
x!{perfkernels_common}
x!{perfkernels_cvtsh_ss_bugfix}
x!{perfkernels_embedding_lookup_avx2}
x!{perfkernels_embedding_lookup_fused_8bit_rowwise_avx2}
x!{perfkernels_embedding_lookup_fused_8bit_rowwise_idx_avx2}
x!{perfkernels_embedding_lookup_idx_avx2}
x!{perfkernels_embedding_lookup_idx}
x!{perfkernels_embedding_lookup}
x!{perfkernels_fused_8bit_rowwise_embedding_lookup_idx}
x!{perfkernels_fused_8bit_rowwise_embedding_lookup}
x!{perfkernels_fused_nbit_rowwise_conversion}
x!{perfkernels_lstm_unit_cpu_avx2}
x!{perfkernels_lstm_unit_cpu_common}
x!{perfkernels_lstm_unit_cpu}
x!{perfkernels_math_cpu_avx2}
x!{perfkernels_math_cpu_base}
x!{perfkernels_math}
x!{perfkernels_typed_axpy_avx2}
x!{perfkernels_typed_axpy_avx}
x!{perfkernels_typed_axpy}
