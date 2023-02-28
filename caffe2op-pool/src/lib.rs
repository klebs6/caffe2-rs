#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_pool_gradient}
x!{op_pool_op_cudnn}
x!{op_pool_op_util}
x!{op_pool}
