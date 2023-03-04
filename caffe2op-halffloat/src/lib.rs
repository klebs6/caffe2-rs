#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{f16_constant_fill}
x!{f16_uniform_fill}
x!{f16_uniform_fill_cpu}
x!{float_to_half}
x!{float_to_half_cpu}
x!{float_to_half_gradient}
x!{half_to_float}
x!{half_to_float_cpu}
x!{half_to_float_gradient}
x!{inference}
x!{test_half}
