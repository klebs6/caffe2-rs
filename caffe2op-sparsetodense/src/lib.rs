#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sparse_to_dense_gradient}
x!{get_sparse_to_dense_mask_gradient}
x!{sparse_to_dense}
x!{sparse_to_dense_config}
x!{sparse_to_dense_mask}
x!{sparse_to_dense_mask_base}
x!{sparse_to_dense_mask_config}
x!{sparse_to_dense_mask_gradient}
x!{sparse_to_dense_mask_gradient_config}
