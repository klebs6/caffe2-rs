#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_softplus_gradient}
x!{softplus}
x!{softplus_gradient}
x!{test_softplus}
