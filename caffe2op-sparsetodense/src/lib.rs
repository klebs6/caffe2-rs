#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_sparse_to_dense_mask}
x!{op_sparse_to_dense}
