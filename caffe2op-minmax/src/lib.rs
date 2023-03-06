#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{max}
x!{max_gradient}
x!{min}
x!{min_gradient}
x!{select_gradient}
x!{test_max}
x!{test_min}
