#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add}
x!{average_pool}
x!{channel_shuffle}
x!{concat}
x!{conv_transpose}
x!{conv}
x!{dequantize}
x!{fc}
x!{flatten}
x!{given_tensor_fill}
x!{leaky_relu}
x!{max_pool}
x!{quantize}
x!{relu}
x!{reshape}
x!{resize_nearest}
x!{roi_align_op_test}
x!{roi_align}
x!{sigmoid}
x!{slice}
x!{softmax}
x!{test_utils}
x!{test_i8quantized}
x!{transpose}
x!{utils}
