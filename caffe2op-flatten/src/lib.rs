#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{flatten}
x!{get_flatten_gradient}
x!{inference}
x!{run_flatten}
x!{test_flatten}
