#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{affine_channel}
x!{affine_channel_gradient}
x!{affine_channel_gradient_cpu}
x!{get_gradient}
x!{nchw}
x!{nhwc}
