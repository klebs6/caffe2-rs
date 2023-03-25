#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{stump_func}
x!{stump_func_cpu}
x!{stump_func_index}
