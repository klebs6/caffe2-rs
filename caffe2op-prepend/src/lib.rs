#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_prepend_dim_gradient}
x!{merge_dim}
x!{merge_dim_config}
x!{prepend_dim}
x!{prepend_dim_config}
