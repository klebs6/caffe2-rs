#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_tanh_gradient}
x!{tanh}
x!{tanh_cpu}
x!{tanh_gradient}
x!{tanh_gradient_cpu}
x!{test_tanh}
