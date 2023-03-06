#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{backward}
x!{compute_internal_gradients}
x!{gamma_beta}
x!{get_gradient}
x!{group_norm}
x!{group_norm_cpu}
x!{group_norm_gradient}
x!{group_norm_gradient_cpu}
x!{group_norm_gradient_run_on_device}
x!{group_norm_run_on_device}
