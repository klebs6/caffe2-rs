#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_reshape_op_gpu_test}
x!{op_reshape}
