#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{opt_annotations}
x!{opt_backend_cutting_test}
x!{opt_backend_cutting}
x!{opt_backend_transformer_base}
x!{opt_bound_shape_inference_test}
x!{opt_bound_shape_inferencer}
x!{opt_converter_nomigraph_test}
x!{opt_converter}
x!{opt_custom_cc_amrc}
x!{opt_custom_concat_elim_test}
x!{opt_custom_concat_elim}
x!{opt_custom_converter_test}
x!{opt_custom_converter}
x!{opt_custom_freeze_quantization_params}
x!{opt_custom_in_batch_broadcast_test}
x!{opt_custom_in_batch_broadcast}
x!{opt_custom_pointwise_elim}
x!{opt_dead_code_elim_test}
x!{opt_dead_code_elim}
x!{opt_device_test}
x!{opt_device}
x!{opt_distributed_converter}
x!{opt_distributed_test}
x!{opt_distributed}
x!{opt_fakefp16_transform}
x!{opt_fusion}
x!{opt_glow_net_transform}
x!{opt_mobile_test}
x!{opt_mobile}
x!{opt_nql_ast}
x!{opt_nql_graphmatcher}
x!{opt_nql_tests_graphmatchertest}
x!{opt_onnx_convert}
x!{opt_onnxifi_transformer}
x!{opt_onnxifi}
x!{opt_optimize_ideep}
x!{opt_optimizer}
x!{opt_passes}
x!{opt_shape_info}
x!{opt_split_slss_test}
x!{opt_tvm_transformer}
