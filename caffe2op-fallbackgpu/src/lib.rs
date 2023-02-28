#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_operator_fallback_gpu_test}
x!{op_operator_fallback_gpu}
