#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{lp_norm}
x!{lp_norm_cpu}
x!{lp_norm_gradient}
x!{lp_norm_gradient_cpu}
x!{test_lp_norm}
