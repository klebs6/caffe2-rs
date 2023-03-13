#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_leaky_relu_gradient}
x!{leaky_relu}
x!{leaky_relu_cpu}
x!{leaky_relu_gradient}
x!{leaky_relu_gradient_cpu}
x!{test_leaky_relu}
