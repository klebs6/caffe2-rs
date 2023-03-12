#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cpu_sparse_lengths_reduction}
x!{doc}
x!{float_to_rowwise_quantized_8bits}
x!{get_tt_sparse_lengths_gradient}
x!{rowwise_8bit_quantized_to_float}
x!{sparse_lengths_8bits_rowwise}
x!{sparse_lengths_fused}
x!{sparse_lengths_fused_config}
x!{sparse_lengths_fused_nbit_rowwise}
x!{sparse_lengths_mean}
x!{sparse_lengths_mean_2bit_rowwise_sparse}
x!{sparse_lengths_mean_8bits_rowwise}
x!{sparse_lengths_mean_gradient}
x!{sparse_lengths_nbit_rowwise_sparse}
x!{sparse_lengths_sum}
x!{sparse_lengths_sum_2bit_rowwise_sparse}
x!{sparse_lengths_sum_4bit_rowwise_sparse}
x!{sparse_lengths_sum_8bit_rowwise_sparse}
x!{sparse_lengths_sum_8bits_rowwise}
x!{sparse_lengths_sum_fused_2bit_rowwise}
x!{sparse_lengths_sum_fused_4bit_rowwise}
x!{sparse_lengths_sum_sparse_lookup}
x!{sparse_lengths_weighted_mean_8bits_rowwise}
x!{sparse_lengths_weighted_sum}
x!{sparse_lengths_weighted_sum_8bits_rowwise}
x!{sparse_lengths_weighted_sum_gradient}
x!{tt_sparse_lengths_sum}
x!{tt_sparse_lengths_sum_gradient}
