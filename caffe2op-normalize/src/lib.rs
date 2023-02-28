#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_norm_planar_yuv}
x!{op_normalize_l1}
x!{op_normalize}
