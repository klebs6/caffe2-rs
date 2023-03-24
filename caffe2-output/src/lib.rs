#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{output_min_max_net_observer}
x!{output_min_max_observer}
x!{register_quantization_params_net_observer}
x!{register_quantization_params_with_histogram_net_observer}
