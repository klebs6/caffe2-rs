#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{lengths_topk}
x!{lengths_topk_gradient}
x!{run_on_device}
