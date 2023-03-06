#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{average_pool_cpu}
x!{average_pool_cudnn}
x!{average_pool_docs}
x!{average_pool_fwd}
x!{average_pool_gradient_register}
x!{average_pool_neon}
x!{average_pool_register}
x!{average_pool}
x!{compute_average_pool_gradient}
x!{compute_average_pool}
x!{compute_max_pool_gradient}
x!{compute_max_pool}
x!{cudnn_pool_gradient_op}
x!{cudnn_pool_op}
x!{get_gradient}
x!{max_pool_cpu}
x!{max_pool_cudnn}
x!{max_pool_docs}
x!{max_pool_forward}
x!{max_pool_gradient_register}
x!{max_pool_neon}
x!{max_pool_register}
x!{max_pool_with_index}
x!{max_pool}
x!{neon}
x!{pool_gradient_op}
x!{pool_op}
x!{run_average_pool_gradient}
x!{run_average_pool}
x!{run_max_pool_gradient}
x!{run_max_pool}
x!{set_tensor_descriptor}
x!{test_average_pool}
x!{test_max_pool}
