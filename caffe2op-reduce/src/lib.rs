#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_reduce_front_back_max_ops}
x!{op_reduce_front_back_mean_ops}
x!{op_reduce_front_back_sum_mean_ops}
x!{op_reduce_front_back_sum_ops}
x!{op_reduce_ops}
x!{op_reducer_functors}
x!{op_reduction_ops}
