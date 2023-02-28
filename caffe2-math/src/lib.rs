#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{util_hip_math_blas_gpu_test}
x!{util_math_broadcast}
x!{util_math_cpu}
x!{util_math_elementwise}
x!{util_math_gpu_test}
x!{util_math_half_utils}
x!{util_math_reduce}
x!{util_math_test}
x!{util_math_transpose}
x!{util_math_utils}
x!{util_math}
