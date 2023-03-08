#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{gather_by_key}
x!{get_gather_by_key_gradient}
x!{lengths_partition}
x!{modulo_partition}
x!{partition}
x!{partition_op_base}
x!{register}
