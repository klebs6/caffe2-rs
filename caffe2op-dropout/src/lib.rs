#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_dropout}
x!{cudnn_dropout_gradient}
x!{dropout}
x!{dropout_cpu}
x!{dropout_gradient}
x!{dropout_gradient_cpu}
x!{test_dropout}
