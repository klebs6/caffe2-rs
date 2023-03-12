#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{compress_uniform_simplified}
x!{convert}
x!{float_to_fused_2bit_rowwise_quantized}
x!{float_to_fused_4bit_rowwise_quantized}
x!{float_to_fused_8bit_rowwise_quantized}
x!{float_to_fused_8bit_rowwise_quantized_halfscale_bias}
x!{float_to_fused_nbit_fake_rowwise_quantized}
x!{float_to_fused_nbit_rowwise_quantized}
x!{float_to_fused_rand_rowwise_quantized}
x!{fused_2bit_rowwise_quantized_to_float}
x!{fused_2bit_rowwise_quantized_to_half}
x!{fused_4bit_rowwise_quantized_to_float}
x!{fused_4bit_rowwise_quantized_to_half}
x!{fused_8bit_rowwise_quantized_halfscale_bias_to_float}
x!{fused_8bit_rowwise_quantized_halfscale_bias_to_half_float}
x!{fused_8bit_rowwise_quantized_to_float}
x!{fused_8bit_rowwise_quantized_to_half_float}
x!{fused_nbit_rowwise_quantized_to_float}
x!{fused_rand_rowwise_quantized_to_float}
x!{half_float_to_fused_8bit_rowwise_quantized}
x!{half_float_to_fused_8bit_rowwise_quantized_halfscale_bias}
x!{half_to_fused_2bit_rowwise_quantized}
x!{half_to_fused_4bit_rowwise_quantized}
x!{param_search_greedy}
x!{register}
x!{run_float_to_fused_8bit_rowwise_quantized}
x!{run_float_to_fused_rand_rowwise_quantized}
x!{run_fused_8bit_rowwise_quantized_to_float}
x!{run_fused_nbit_rowwise_quantized_to_float}
x!{run_fused_rand_rowwise_quantized_to_float}
