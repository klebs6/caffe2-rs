#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{config}
x!{f16_cpu}
x!{f32_cpu}
x!{sparse_normalize}
