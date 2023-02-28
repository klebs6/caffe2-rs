#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_local_response_normalization_op_cudnn}
x!{op_local_response_normalization}
