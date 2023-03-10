#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{server_quantize_activation_distribution_observer}
x!{server_quantize_batch_matmul_dnnlowp}
x!{server_quantize_batch_permutation_dnnlowp}
x!{server_quantize_caffe2_dnnlowp_utils}
x!{server_quantize_channel_shuffle_dnnlowp}
x!{server_quantize_compute_equalization_scale}
x!{server_quantize_concat_dnnlowp}
x!{server_quantize_conv_dnnlowp_acc16}
x!{server_quantize_conv_dnnlowp}
x!{server_quantize_conv_pool_dnnlowp_op_base}
x!{server_quantize_conv_relu}
x!{server_quantize_dequantize_dnnlowp}
x!{server_quantize_dnnlowp_partition}
x!{server_quantize_dnnlowp}
x!{server_quantize_dynamic_histogram_test}
x!{server_quantize_dynamic_histogram}
x!{server_quantize_elementwise_add_dnnlowp}
x!{server_quantize_elementwise_dnnlowp}
x!{server_quantize_elementwise_linear_dnnlowp}
x!{server_quantize_elementwise_mul_dnnlowp}
x!{server_quantize_elementwise_sum_benchmark}
x!{server_quantize_elementwise_sum_dnnlowp_op_avx2}
x!{server_quantize_elementwise_sum_dnnlowp}
x!{server_quantize_elementwise_sum_relu}
x!{server_quantize_fb_fc_packed}
x!{server_quantize_fbgemm_fp16_pack}
x!{server_quantize_fbgemm_pack_blob}
x!{server_quantize_fbgemm_pack_matrix_cache}
x!{server_quantize_fbgemm_pack}
x!{server_quantize_fc_fake_lowp_test}
x!{server_quantize_fully_connected_dnnlowp_acc16}
x!{server_quantize_fully_connected_dnnlowp}
x!{server_quantize_fully_connected_fake_lowp_op_avx2}
x!{server_quantize_fully_connected_fake_lowp}
x!{server_quantize_group_norm_dnnlowp_op_avx2}
x!{server_quantize_group_norm_dnnlowp}
x!{server_quantize_im2col_dnnlowp}
x!{server_quantize_int8_gen_quant_params_min_max}
x!{server_quantize_int8_gen_quant_params}
x!{server_quantize_int8_quant_scheme_blob_fill}
x!{server_quantize_kl_minimization_example}
x!{server_quantize_kl_minimization}
x!{server_quantize_l1_minimization_example}
x!{server_quantize_l2_minimization_approx_example}
x!{server_quantize_l2_minimization_example}
x!{server_quantize_l2_minimization_test}
x!{server_quantize_l2_minimization}
x!{server_quantize_lstm_unit_dnnlowp}
x!{server_quantize_mmio}
x!{server_quantize_norm_minimization_avx2}
x!{server_quantize_norm_minimization}
x!{server_quantize_op_wrapper}
x!{server_quantize_p99_example}
x!{server_quantize_pool_dnnlowp_op_avx2}
x!{server_quantize_pool_dnnlowp}
x!{server_quantize_quantization_error_minimization}
x!{server_quantize_quantize_dnnlowp}
x!{server_quantize_relu_dnnlowp_op_avx2}
x!{server_quantize_relu_dnnlowp}
x!{server_quantize_requantization_test}
x!{server_quantize_resize_nearest_3d_dnnlowp}
x!{server_quantize_resize_nearest_dnnlowp}
x!{server_quantize_sigmoid_dnnlowp}
x!{server_quantize_sigmoid_test}
x!{server_quantize_sigmoid}
x!{server_quantize_spatial_batch_norm_dnnlowp_op_avx2}
x!{server_quantize_spatial_batch_norm_dnnlowp}
x!{server_quantize_spatial_batch_norm_relu}
x!{server_quantize_tanh_dnnlowp}
x!{server_quantize_tanh_test}
x!{server_quantize_tanh}
x!{server_quantize_transpose}
x!{server_quantize_utility_dnnlowp_ops}
