#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{replace_nan}
x!{replace_nan_cpu}
x!{run_on_device}
