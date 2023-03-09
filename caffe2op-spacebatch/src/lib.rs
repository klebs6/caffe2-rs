#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{batch_to_space}
x!{get_batch_to_space_gradient}
x!{get_space_to_batch_gradient}
x!{space_batch_op_base}
x!{space_to_batch}
x!{test_batch_to_space}
x!{test_space_to_batch}
