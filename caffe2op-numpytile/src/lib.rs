#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{numpy_tile}
x!{run_on_device}
