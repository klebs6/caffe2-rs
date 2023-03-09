#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_softmax_with_loss_gradient}
x!{spatial_softmax_with_loss}
x!{spatial_softmax_with_loss_config}
x!{spatial_softmax_with_loss_cpu}
x!{spatial_softmax_with_loss_gradient}
x!{spatial_softmax_with_loss_gradient_cpu}
