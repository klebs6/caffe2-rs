#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{eigen_pow_functor}
x!{get_pow_gradient}
x!{pow}
x!{pow_config}
x!{pow_create}
x!{pow_run}
x!{test_pow}
