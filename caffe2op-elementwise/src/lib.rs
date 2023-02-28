#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_elementwise_add}
x!{op_elementwise_div_gradient}
x!{op_elementwise_div}
x!{op_elementwise_linear}
x!{op_elementwise_logical_ops}
x!{op_elementwise_mul}
x!{op_elementwise_op_gpu_test}
x!{op_elementwise_op_test}
x!{op_elementwise_ops_schema}
x!{op_elementwise_ops_utils}
x!{op_elementwise_ops}
x!{op_elementwise_sub}
x!{op_elementwise_sum}
