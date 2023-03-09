#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{upsample_bilinear}
x!{upsample_bilinear_cpu}
x!{upsample_bilinear_gradient}
x!{upsample_bilinear_gradient_cpu}
