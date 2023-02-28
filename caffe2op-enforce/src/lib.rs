#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_enforce_finite}
x!{op_ensure_clipped}
x!{op_ensure_cpu_output}
