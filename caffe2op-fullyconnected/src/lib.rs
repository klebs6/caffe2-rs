#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{fully_connected}
x!{fully_connected_gradient}
x!{fully_connected_op_gpu}
x!{fully_connected_transpose}
x!{get_fully_connected_gradient}
x!{inference}
x!{register}
x!{run_fully_connected}
x!{run_fully_connected_gradient}
x!{run_on_cuda}
x!{test_fully_connected}
