#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_quantized_int8_add}
x!{op_quantized_int8_average_pool}
x!{op_quantized_int8_channel_shuffle}
x!{op_quantized_int8_concat}
x!{op_quantized_int8_conv_transpose}
x!{op_quantized_int8_conv}
x!{op_quantized_int8_dequantize}
x!{op_quantized_int8_fc}
x!{op_quantized_int8_flatten}
x!{op_quantized_int8_given_tensor_fill}
x!{op_quantized_int8_leaky_relu}
x!{op_quantized_int8_max_pool}
x!{op_quantized_int8_quantize}
x!{op_quantized_int8_relu}
x!{op_quantized_int8_reshape}
x!{op_quantized_int8_resize_nearest}
x!{op_quantized_int8_roi_align_op_test}
x!{op_quantized_int8_roi_align}
x!{op_quantized_int8_sigmoid}
x!{op_quantized_int8_slice}
x!{op_quantized_int8_softmax}
x!{op_quantized_int8_test_utils}
x!{op_quantized_int8_test}
x!{op_quantized_int8_transpose}
x!{op_quantized_int8_utils}
