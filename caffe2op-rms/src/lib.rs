#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{rms_norm}
x!{rms_norm_cpu}
x!{rms_norm_gradient}
x!{rms_norm_gradient_cpu}
