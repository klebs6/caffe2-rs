#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{selu}
x!{selu_gradient}
x!{selu_gradient_cpu}
x!{test_selu}
