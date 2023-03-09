#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_spatial_batch_norm_gradient}
x!{inference}
x!{spatial_batch_norm}
x!{spatial_batch_norm_config}
x!{spatial_batch_norm_gradient}
x!{spatial_batch_norm_gradient_config}
x!{spatial_batch_norm_gradient_cpu}
x!{spatial_batch_norm_inference}
