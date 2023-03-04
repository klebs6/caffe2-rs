#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{col2im}
x!{col2im_gradient}
x!{im2col}
x!{im2col_gradient}
x!{im2col_tensor_inference}
