#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{exp}
x!{exp_gradient}
x!{test_exp}
