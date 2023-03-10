#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{arg}
x!{arg_max_reducer}
x!{arg_max_reducer_cpu}
x!{arg_min_reducer}
x!{arg_min_reducer_cpu}
x!{compute}
x!{infer}
x!{test_arg_max_reducer}
x!{test_arg_min_reducer}
