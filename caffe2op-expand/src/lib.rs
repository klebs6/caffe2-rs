#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{expand}
x!{expand_dims}
x!{expand_gradient}
x!{get_expand_gradient}
x!{get_gradient}
x!{register}
x!{run_expand_dims}
x!{run_squeeze}
x!{squeeze}
x!{test_expand_dims}
x!{test_squeeze}
