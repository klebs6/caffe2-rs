#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{inference}
x!{matmul}
x!{run_on_device}
x!{test_matmul}
