#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_fused_rowwise_8bit_conversion_ops}
x!{op_fused_rowwise_nbit_conversion_ops}
x!{op_fused_rowwise_nbitfake_conversion_ops}
x!{op_fused_rowwise_random_quantization_ops}
