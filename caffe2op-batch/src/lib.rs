#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{batch_box_cox}
x!{batch_bucketize}
x!{batch_dense_to_sparse}
x!{batch_dense_to_sparse_cpu}
x!{batch_dense_to_sparse_gradient}
x!{batch_gather}
x!{batch_gather_gradient}
x!{batch_matmul}
x!{batch_matmul_gradient}
x!{batch_matmul_inference}
x!{batch_moments}
x!{batch_moments_cpu}
x!{batch_moments_gradient}
x!{batch_moments_gradient_cpu}
x!{batch_permutation}
x!{batch_permutation_cpu}
x!{batch_permutation_gradient}
x!{batch_permutation_loop}
x!{batch_sparse_to_dense}
x!{batch_sparse_to_dense_cpu}
x!{batch_sparse_to_dense_gradient}
x!{cached_buffers}
x!{concat_batch_matmul_batch_gather}
x!{delegate}
x!{test_batch_matmul}
x!{test_batch_matmul_gpu}
x!{test_batch_permutation_gpu}
x!{tile_indices_array}
