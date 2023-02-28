#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_conv_gradient}
x!{op_conv_op_cache_cudnn_test}
x!{op_conv_op_cache_cudnn}
x!{op_conv_op_cudnn}
x!{op_conv_op_eigen}
x!{op_conv_op_gpu}
x!{op_conv_op_impl}
x!{op_conv_op_shared}
x!{op_conv_pool_op_base}
x!{op_conv_transpose_gradient}
x!{op_conv_transpose_op_cudnn}
x!{op_conv_transpose_op_impl}
x!{op_conv_transpose_op_mobile_impl}
x!{op_conv_transpose_op_mobile_test}
x!{op_conv_transpose_op_mobile}
x!{op_conv_transpose_unpool_op_base}
x!{op_conv_transpose}
x!{op_conv}
