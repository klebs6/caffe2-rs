#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_tan_gradient}
x!{get_tanh_gradient}
x!{tan_gradient}
x!{tanh_cpu}
x!{tanh_gradient_cpu}
x!{tanh_gradient}
x!{tanh}
x!{tan}
x!{test_tanh}
