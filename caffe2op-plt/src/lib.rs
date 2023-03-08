#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{check_bounds}
x!{config}
x!{get_param_data}
x!{infer}
x!{op}
x!{piecewise_linear_transform}
x!{run_on_device}
x!{transform_binary}
x!{transform_general}
