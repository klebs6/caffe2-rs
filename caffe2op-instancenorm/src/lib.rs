#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{compute_fused_params}
x!{gamma_beta_backward}
x!{get_gradient}
x!{instance_norm}
x!{instance_norm_cpu}
x!{instance_norm_gradient}
x!{instance_norm_gradient_cpu}
x!{nchw}
x!{nhwc}
x!{test_instance_norm}
