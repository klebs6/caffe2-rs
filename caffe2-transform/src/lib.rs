#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_transform_test}
x!{core_transform}
x!{txform_common_subexpression_elimination_test}
x!{txform_common_subexpression_elimination}
x!{txform_conv_to_nnpack_transform_test}
x!{txform_conv_to_nnpack_transform}
x!{txform_pattern_net_transform_test}
x!{txform_pattern_net_transform}
x!{txform_single_op_transform}
