#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_locally_connected_op_impl}
x!{op_locally_connected_op_util}
x!{op_locally_connected}
