#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_common_cudnn}
x!{core_common_gpu}
x!{core_common_test}
x!{core_common}
