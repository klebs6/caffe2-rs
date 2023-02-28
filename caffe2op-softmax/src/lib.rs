#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_softmax_op_cudnn}
x!{op_softmax_utils}
x!{op_softmax_with_loss}
x!{op_softmax}
