#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{shape}
x!{run_on_device}
x!{test_shape}
