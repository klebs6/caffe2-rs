#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{config}
x!{get_gradient}
x!{inference}
x!{reshape}
x!{run_on_device}
x!{test_reshape}
