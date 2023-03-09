#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{rsqrt}
x!{rsqrt_cpu}
x!{rsqrt_gradient}
