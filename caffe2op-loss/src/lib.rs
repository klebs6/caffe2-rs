#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{averaged_loss}
x!{averaged_loss_gradient}
x!{get_gradient}
x!{test_averaged_loss}
