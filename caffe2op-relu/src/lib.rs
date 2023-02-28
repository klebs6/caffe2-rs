#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_relu_n}
x!{op_relu}
