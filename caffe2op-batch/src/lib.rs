#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_batch_box_cox}
x!{op_batch_bucketize}
x!{op_batch_gather_ops}
x!{op_batch_matmul_op_gpu_test}
x!{op_batch_matmul_op_test}
x!{op_batch_matmul}
x!{op_batch_moments}
x!{op_batch_permutation_op_gpu_test}
x!{op_batch_permutation}
x!{op_batch_sparse_to_dense}
x!{op_cc_bmm_bg}
