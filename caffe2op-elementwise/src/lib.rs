#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add}
x!{binary_elementwise}
x!{binary_elementwise_gradient}
x!{binary_functor}
x!{broadcast}
x!{compute_div_gradient}
x!{compute_mul_gradient}
x!{div}
x!{div_cpu}
x!{div_gradient}
x!{doc}
x!{elementwise_sum}
x!{forward_only_binary_functor}
x!{get_add_gradient}
x!{get_elementwise_linear_gradient}
x!{get_mul_gradient}
x!{gpu_test}
x!{inference}
x!{is_member_of}
x!{is_member_of_value_holder}
x!{linear}
x!{linear_cpu}
x!{linear_gradient}
x!{linear_gradient_cpu}
x!{mul}
x!{mul_cpu}
x!{not}
x!{register}
x!{schema}
x!{sign}
x!{sub}
x!{sub_gradient}
x!{sum_reduce_like}
x!{sum_reduce_like_cpu}
x!{test_add}
x!{test_cmp}
x!{test_div}
x!{test_elementwise_sub}
x!{test_elementwise_sum}
x!{test_is_member_of}
x!{test_linear}
x!{test_logical}
x!{test_mul}
x!{test_not}
x!{test_ops}
x!{test_sign}
x!{unary}
x!{unary_elementwise}
x!{where_op}
