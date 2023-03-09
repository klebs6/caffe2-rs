#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{roi_pool}
x!{roi_pool_cpu}
x!{roi_pool_gradient}
