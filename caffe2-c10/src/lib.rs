#![feature(allocator_api)]
#![feature(specialization)]
#![feature(const_intoiterator_identity)]
#![feature(const_mut_refs)]
#![feature(adt_const_params)]
#![feature(const_for)]
#![feature(const_cmp)]
#![feature(const_trait_impl)]
#![feature(derive_const)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{allocator}
x!{backend}
x!{command_line}
x!{copy_bytes}
x!{cpu_allocator}
x!{cuda_cuda_caching_allocator}
x!{cuda_cuda_exception}
x!{cuda_cuda_functions}
x!{cuda_cuda_graphs_c10utils}
x!{cuda_cuda_guard}
x!{cuda_cuda_macros}
x!{cuda_cuda_math_compat}
x!{cuda_cuda_stream}
x!{cuda_impl__cuda_guard_impl}
x!{cuda_impl__cuda_test}
x!{cuda_test_impl__cuda_test}
x!{default_dtype}
x!{default_tensor_options}
x!{device}
x!{device_guard}
x!{device_type}
x!{dispatch_key}
x!{dispatch_key_set}
x!{event}
x!{generator_impl}
x!{grad_mode}
x!{impl__device_guard_impl_interface}
x!{impl__fake_guard_impl}
x!{impl__inline_device_guard}
x!{impl__inline_event}
x!{impl__inline_stream_guard}
x!{impl__local_dispatch_key_set}
x!{impl__sizes_and_strides}
x!{impl__virtual_guard_impl}
x!{inference_mode}
x!{layout}
x!{macros_export}
x!{macros_macros}
x!{memory_format}
x!{mobile_cpu_caching_allocator}
x!{mobile_cpu_profiling_allocator}
x!{q_engine}
x!{q_scheme}
x!{qint}
x!{scalar}
x!{scalar_type}
x!{scalar_type_to_type_meta}
x!{storage}
x!{storage_impl}
x!{stream}
x!{stream_guard}
x!{tensor_impl}
x!{tensor_options}
x!{test_device_guard_test}
x!{test_dispatch_key_set_test}
x!{test_impl__inline_device_guard_test}
x!{test_impl__inline_stream_guard_test}
x!{test_impl__sizes_and_strides_test}
x!{test_util_accumulate_test}
x!{test_util_bfloat16_test}
x!{test_util_bitset_test}
x!{test_util_constexpr_crc_test}
x!{test_util_exception_test}
x!{test_util_left_right_test}
x!{test_util_logging_test}
x!{test_util_registry_test}
x!{test_util_tempfile_test}
x!{test_util_thread_local_test}
x!{test_util_typeid_test}
x!{thread_pool}
x!{undefined_tensor_impl}
x!{util_accumulate}
x!{util_backtrace}
x!{util_bitset}
x!{util_constexpr_crc}
x!{util_copysign}
x!{util_deadlock_detection}
x!{util_env}
x!{util_exception}
x!{util_llvm_math_extras}
x!{util_logging}
x!{util_logging_is_google_glog}
x!{util_logging_is_not_google_glog}
x!{util_math_compat}
x!{util_numa}
x!{util_registry}
x!{util_scope_exit}
x!{util_signal_handler}
x!{util_small_buffer}
x!{util_string_util}
x!{util_tempfile}
x!{util_thread_local_debug_info}
x!{util_thread_name}
x!{util_ty}
x!{util_type_cast}
x!{util_type_index}
x!{util_type_list}
x!{util_type_traits}
x!{util_typeid}
x!{util_unicode}
x!{util_unique_void_ptr}
x!{util_unroll}
x!{util_win32_headers}
x!{wrap_dim_minimal}


// @note
// 
// the following c++ modules have been deleted (among a few others):
//
// deleted: caffe2-c10/src/macros_cmake_macros.rs
// deleted: caffe2-c10/src/make_lists.rs
// deleted: caffe2-c10/src/test_util_complex_test.rs
// deleted: caffe2-c10/src/test_util_macros.rs
// deleted: caffe2-c10/src/util_align_of.rs
// deleted: caffe2-c10/src/util_array.rs
// deleted: caffe2-c10/src/util_array_ref.rs
// deleted: caffe2-c10/src/util_c++17.rs
// deleted: caffe2-c10/src/util_complex.rs
// deleted: caffe2-c10/src/util_complex_math.rs
// deleted: caffe2-c10/src/util_complex_utils.rs
// deleted: caffe2-c10/src/util_deprecated.rs
// deleted: caffe2-c10/src/util_either.rs
// deleted: caffe2-c10/src/util_exclusively_owned.rs
// deleted: caffe2-c10/src/util_flags.rs
// deleted: caffe2-c10/src/util_flags_use_gflags.rs
// deleted: caffe2-c10/src/util_flags_use_no_gflags.rs
// deleted: caffe2-c10/src/util_flat_hash_map.rs
// deleted: caffe2-c10/src/util_float16.rs
// deleted: caffe2-c10/src/util_float16_inl.rs
// deleted: caffe2-c10/src/util_float16_math.rs
// deleted: caffe2-c10/src/util_function_ref.rs
// deleted: caffe2-c10/src/util_half.rs
// deleted: caffe2-c10/src/util_half_inl.rs
// deleted: caffe2-c10/src/util_hash.rs
// deleted: caffe2-c10/src/util_id_wrapper.rs
// deleted: caffe2-c10/src/util_in_place.rs
// deleted: caffe2-c10/src/util_intrusive_ptr.rs
// deleted: caffe2-c10/src/util_irange.rs
// deleted: caffe2-c10/src/util_left_right.rs
// deleted: caffe2-c10/src/util_math_constants.rs
// deleted: caffe2-c10/src/util_maybe_owned.rs
// deleted: caffe2-c10/src/util_metaprogramming.rs
// deleted: caffe2-c10/src/util_optional.rs
// deleted: caffe2-c10/src/util_order_preserving_flat_hash_map.rs
// deleted: caffe2-c10/src/util_overloaded.rs
// deleted: caffe2-c10/src/util_python_stub.rs
// deleted: caffe2-c10/src/util_reverse_iterator.rs
// deleted: caffe2-c10/src/util_small_vector.rs
// deleted: caffe2-c10/src/util_sparse_bitset.rs
// deleted: caffe2-c10/src/util_string_utils.rs
// deleted: caffe2-c10/src/util_string_view.rs
// deleted: caffe2-c10/src/util_thread_local.rs
//
// generally, the reason is because something in
// rust already implements the contained ideas
//
// It is possible there is some level of
// incompatibility, but the goal should be to get
// the system to work without relying on homemade
// duplicate functionality (to Cow, SmallVec,
// Option, etc)
//
