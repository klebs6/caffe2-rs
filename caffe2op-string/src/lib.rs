#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{config}
x!{for_each}
x!{starts_with}
x!{string_equals}
x!{string_join}
x!{string_join_cpu}
x!{test_string_ops}
