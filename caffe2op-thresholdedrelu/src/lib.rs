#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_thresholded_relu_gradient}
x!{thresholded_relu}
x!{thresholded_relu_cpu}
x!{thresholded_relu_gradient}
x!{thresholded_relu_gradient_cpu}
