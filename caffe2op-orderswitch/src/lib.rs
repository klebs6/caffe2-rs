#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_order_switch_ops_cudnn}
x!{op_order_switch_ops}
