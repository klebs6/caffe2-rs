#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_layer_norm_gradient}
x!{layer_norm}
x!{layer_norm_cpu}
x!{layer_norm_gradient}
x!{layer_norm_gradient_cpu}
x!{run_layer_norm}
