#![feature(specialization)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{build_options}
x!{caffe_config}
x!{caffe_cuda_set_device}
x!{core_macros}
x!{core_static_tracepoint}
x!{core_static_tracepoint_elfx86}
x!{core_test_utils}
x!{core_types}
x!{cublas_enforce}
x!{cublas_get_error_string}
x!{cuda_config}
x!{cuda_device_prop_wrapper}
x!{cuda_enforce}
x!{cuda_kernel_loop}
x!{cuda_runtime_flag_flipper}
x!{cuda_version}
x!{cudnn_enforce}
x!{cudnn_errors}
x!{cudnn_filter_desc_wrapper}
x!{cudnn_tensor_desc_wrapper}
x!{cudnn_tensor_format}
x!{cudnn_type_wrapper}
x!{cudnn_version}
x!{curand_enforce}
x!{curand_get_error_string}
x!{device_query}
x!{dispatch_function_by_value_with_type}
x!{dynamic_cast}
x!{get_cuda_peer_access_pattern}
x!{get_default_gpuid}
x!{get_device_property}
x!{get_gpuid_for_pointer}
x!{num_cuda_devices}
x!{runtime}
x!{simple_array}
x!{skip_indices}
x!{tensor_core_available}
x!{test_stoid}
x!{type_traits}
