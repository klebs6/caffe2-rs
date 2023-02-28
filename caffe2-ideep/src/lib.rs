#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{ideep_ideep_utils}
x!{ideep_operators_adam}
x!{ideep_operators_channel_shuffle}
x!{ideep_operators_concat_split}
x!{ideep_operators_conv_pool_base}
x!{ideep_operators_conv_transpose_unpool_base}
x!{ideep_operators_conv_transpose}
x!{ideep_operators_conv}
x!{ideep_operators_dropout}
x!{ideep_operators_elementwise_sum}
x!{ideep_operators_expand_squeeze_dims}
x!{ideep_operators_fully_connected}
x!{ideep_operators_local_response_normalization}
x!{ideep_operators_momentum_sgd}
x!{ideep_operators_operator_fallback_ideep}
x!{ideep_operators_order_switch_ops}
x!{ideep_operators_pool}
x!{ideep_operators_quantization_int8_add}
x!{ideep_operators_quantization_int8_conv}
x!{ideep_operators_quantization_int8_dequantize}
x!{ideep_operators_quantization_int8_fully_connected}
x!{ideep_operators_quantization_int8_given_tensor_fill}
x!{ideep_operators_quantization_int8_pool}
x!{ideep_operators_quantization_int8_quantize}
x!{ideep_operators_quantization_int8_relu}
x!{ideep_operators_queue_ops}
x!{ideep_operators_relu}
x!{ideep_operators_reshape}
x!{ideep_operators_shape}
x!{ideep_operators_sigmoid}
x!{ideep_operators_spatial_batch_norm}
x!{ideep_operators_transpose}
x!{ideep_operators_utility_ops}
x!{ideep_utils_ideep_context}
x!{ideep_utils_ideep_operator}
x!{ideep_utils_ideep_register}
