#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_activation}
x!{cudnn_activation_base}
x!{cudnn_activation_gradient}
