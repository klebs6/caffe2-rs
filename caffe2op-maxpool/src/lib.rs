#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_max_pool_with_index_gpu}
