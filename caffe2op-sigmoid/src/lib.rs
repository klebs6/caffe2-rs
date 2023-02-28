#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_sigmoid_gradient}
x!{op_sigmoid}
