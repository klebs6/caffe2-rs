#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_fully_connected_op_gpu}
x!{op_fully_connected}
x!{inference}
