#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{prelu}
x!{prelu_config}
x!{prelu_cpu}
x!{prelu_gradient}
x!{prelu_gradient_cpu}
x!{prelu_neon}
x!{test_prelu}
