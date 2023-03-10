#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{concat}
x!{concat_op_dev_infer}
x!{cost_inference_for_concat}
x!{get_gradient}
x!{register}
x!{run_on_device}
x!{split}
x!{split_by_lengths}
x!{split_op_dev_infer}
x!{tensor_inference_for_concat}
x!{tensor_inference_for_split}
x!{test_concat_split}
