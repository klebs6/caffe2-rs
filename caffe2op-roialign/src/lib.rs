#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_roi_align_gradient}
x!{op_roi_align_op_gpu_test}
x!{op_roi_align_rotated_gradient}
x!{op_roi_align_rotated}
x!{op_roi_align}
