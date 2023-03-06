#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_lrn}
x!{cudnn_lrn_gradient}
x!{get_gradient}
x!{lrn}
x!{lrn_base}
x!{lrn_cpu}
x!{lrn_gradient}
x!{lrn_gradient_cpu}
x!{test_lrn}
