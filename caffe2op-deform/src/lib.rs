#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{deform_conv_op}
x!{deform_conv_op_base}
x!{deform_conv_op_gradient}
x!{get_gradient_defs}
