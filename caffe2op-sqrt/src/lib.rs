#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sqrt_gradient}
x!{get_square_root_divide_gradient}
x!{sqrt_functor}
x!{square_root_divide}
x!{test_sqrt_functor}
