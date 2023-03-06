#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{gelu_cpu}
x!{gelu_gradient}
x!{gelu_gradient_cpu}
x!{get_gradient}
x!{inference}
x!{gelu}
