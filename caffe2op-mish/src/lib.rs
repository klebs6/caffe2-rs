#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{mish}
x!{mish_gradient}
x!{mish_gradient_cpu}

