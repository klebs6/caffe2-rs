#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_normalize_gradient}
x!{l1}
x!{l2}
x!{l2_gradient}
x!{normalize_planar_yuv}
