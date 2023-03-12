#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cosh}
x!{cosh_gradient}
x!{get_cosh_gradient}
x!{register}
x!{test_cosh}
