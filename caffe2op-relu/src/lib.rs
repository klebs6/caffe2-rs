#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cpu}
x!{get_gradient}
x!{inference}
x!{relu_functor}
x!{relu_gradient_functor}
x!{relun_functor}
x!{relun_gradient_functor}
x!{test_relu}
