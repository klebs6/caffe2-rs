#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{mean}
x!{mean_gradient}
x!{test_mean}
