#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{lp_pool}
x!{lp_pool_gradient}
x!{op}
x!{test_lp_pool}
