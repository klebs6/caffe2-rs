#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cpu_gradient}
x!{cpu_op_resize}
x!{cpu_op_resize_3d}
x!{cpu_op_resize_3d_gradient}
x!{get_gradient}
x!{get_gradient_3d}
x!{gradient}
x!{gradient_3d}
x!{resize_nearest}
x!{resize_nearest_3d}
x!{resize_nearest_3dnchw2x}
x!{resize_nearest_nchw2x}
