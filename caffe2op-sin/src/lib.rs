#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sin_gradient}
x!{get_sinh_gradient}
x!{sin_functor}
x!{sin_gradient_functor}
x!{sin_gradient}
x!{sinh_gradient}
x!{sinh}
x!{sinusoid_position_encoding}
x!{sin}
x!{test_sinh}
x!{test_sin}
