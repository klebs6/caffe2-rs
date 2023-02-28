#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{register_quantization_params_net_observer}
x!{register_quantization_params_with_histogram_net_observer}
