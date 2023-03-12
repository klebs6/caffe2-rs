#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_activation_elu}
x!{cudnn_activation_gradient_elu}
x!{elu}
x!{elu_cpu}
x!{elu_gradient}
x!{elu_gradient_cpu}
x!{get_elu_gradient}
x!{test_elu}
