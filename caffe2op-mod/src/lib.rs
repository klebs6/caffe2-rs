#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_mod}
x!{run_on_cpu}
x!{run_on_device}
x!{test_mod}
