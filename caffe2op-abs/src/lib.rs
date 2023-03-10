#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{abs}
x!{abs_gradient}
x!{get_gradient}
x!{test_abs}
