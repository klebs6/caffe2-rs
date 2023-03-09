#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_swish_gradient}
x!{swish}
x!{swish_cpu}
x!{swish_gradient}
x!{swish_gradient_cpu}
