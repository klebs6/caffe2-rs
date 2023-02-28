#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_lengths_reducer_fused_8bit_rowwise_ops}
x!{op_lengths_reducer_fused_nbit_rowwise_ops}
x!{op_lengths_reducer_ops}
x!{op_lengths_reducer_rowwise_8bit_ops}
