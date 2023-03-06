#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{logit}
x!{logit_cpu}
x!{logit_gradient}
x!{logit_gradient_cpu}
