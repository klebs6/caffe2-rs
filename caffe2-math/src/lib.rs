#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{affine_channel}
x!{axpy_impl}
x!{basic}
x!{broadcast_gpu}
x!{check_reduce_dims}
x!{compute_transpose_axes_for_reduce_op}
x!{compute_transpose_axes_for_reduce_op_with_reduce_axes}
x!{cpu_math}
x!{elementwise}
x!{execute_gpu}
x!{gemm_batched_gpu}
x!{get_index_from_dims}
x!{half_add}
x!{half_div}
x!{half_mul}
x!{half_sub}
x!{increase_index_in_dims}
x!{is_batch_transpose_2d}
x!{is_both_ends_broadcast_binary}
x!{is_both_ends_reduce}
x!{is_identity_permutation}
x!{is_rowwise_broadcast_binary}
x!{is_rowwise_reduce}
x!{math}
x!{reduce}
x!{scale_impl}
x!{specialized_compute_broadcast_binary}
x!{specialized_compute_transpose_stride}
x!{test_broadcast_gpu}
x!{test_gpu}
x!{test_hip_math}
x!{test_math}
x!{transpose}
