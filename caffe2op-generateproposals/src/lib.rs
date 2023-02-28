#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_generate_proposals_op_gpu_test}
x!{op_generate_proposals_op_test}
x!{op_generate_proposals_op_util_boxes_test}
x!{op_generate_proposals_op_util_boxes}
x!{op_generate_proposals_op_util_nms_gpu_test}
x!{op_generate_proposals_op_util_nms_gpu}
x!{op_generate_proposals_op_util_nms_test}
x!{op_generate_proposals_op_util_nms}
x!{op_generate_proposals}
