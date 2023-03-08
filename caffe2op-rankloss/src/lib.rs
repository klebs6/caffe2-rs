#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{log_logit}
x!{pairwise_loss}
x!{pairwise_loss_config}
x!{pairwise_loss_gradient}
x!{pairwise_loss_gradient_config}
