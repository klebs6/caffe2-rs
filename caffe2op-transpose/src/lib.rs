#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_transpose}
x!{get_gradient}
x!{run_on_device}
x!{test_transpose}
x!{inference}
x!{transpose}
