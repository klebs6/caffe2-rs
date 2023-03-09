#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sigmoid_gradient}
x!{sigmoid}
x!{sigmoid_cpu}
x!{sigmoid_gradient}
x!{sigmoid_gradient_cpu}
x!{test_sigmoid}
