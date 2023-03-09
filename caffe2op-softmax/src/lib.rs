#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_softmax}
x!{cudnn_softmax_gradient}
x!{get_softmax_gradient}
x!{get_softmax_with_loss_gradient}
x!{softmax}
x!{softmax_cpu}
x!{softmax_gradient}
x!{softmax_gradient_cpu}
x!{softmax_with_loss}
x!{softmax_with_loss_cpu}
x!{softmax_with_loss_gradient}
x!{softmax_with_loss_gradient_cpu}
x!{specialized_softmax}
x!{test_softmax}
