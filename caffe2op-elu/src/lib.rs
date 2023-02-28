#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_elu_op_cudnn}
x!{op_elu}
