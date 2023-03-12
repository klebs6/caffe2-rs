#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add_input}
x!{algorithms_cache}
x!{compare}
x!{conv}
x!{conv_doc}
x!{conv_gradient}
x!{conv_pool_base}
x!{conv_register}
x!{conv_transpose}
x!{conv_transpose_gradient}
x!{conv_transpose_mobile}
x!{conv_transpose_unpool}
x!{cost_inference}
x!{cudnn_conv}
x!{cudnn_conv_base}
x!{cudnn_conv_gradient}
x!{cudnn_conv_op_base}
x!{cudnn_conv_transpose}
x!{cudnn_conv_transpose_base}
x!{cudnn_conv_transpose_gradient}
x!{eigen_conv}
x!{get_conv_gradient}
x!{get_conv_transpose_gradient}
x!{register_cpu}
x!{register_cuda}
x!{register_cudnn}
x!{register_gradient}
x!{reinterleave_multi}
x!{reinterleave_rows}
x!{run_conv_gradient_on_device}
x!{run_conv_on_device}
x!{run_conv_transpose}
x!{run_conv_transpose_gradient}
x!{run_conv_transpose_mobile}
x!{run_tile}
x!{shared_buffer}
x!{tensor_inference}
x!{test_algorithm_cache}
x!{test_conv}
x!{test_conv_transpose}
x!{test_conv_transpose_mobile}
