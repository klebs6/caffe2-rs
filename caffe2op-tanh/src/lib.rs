#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_tanh_gradient}
x!{op_tanh}
