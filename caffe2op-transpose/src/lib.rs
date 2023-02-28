#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_transpose_op_cudnn}
x!{op_transpose}
