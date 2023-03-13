#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{no_default_engine}
x!{op_utility_ops}
x!{op_utility_ops_gpu_test}
x!{op_utility_ops_test}
x!{op_utils_cudnn}
