#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{assert}
x!{run_on_device}
x!{test_assert}
