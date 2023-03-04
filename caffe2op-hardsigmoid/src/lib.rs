#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{hard_sigmoid}
x!{hard_sigmoid_cpu}
x!{hard_sigmoid_cost_inference}
x!{hard_sigmoid_gradient}
x!{hard_sigmoid_gradient_cpu}
x!{test_hard_sigmoid}
x!{get_gradient}
