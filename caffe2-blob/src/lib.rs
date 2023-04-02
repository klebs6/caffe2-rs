#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{blob}
x!{blob_gpu_test}
x!{blob_stats}
x!{blob_test}
x!{tensor_stat_getter}
