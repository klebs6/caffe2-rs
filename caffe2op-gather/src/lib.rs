#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_gather_fused_8bit_rowwise}
x!{op_gather_ranges_to_dense}
x!{op_gather}
