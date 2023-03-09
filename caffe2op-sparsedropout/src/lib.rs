#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{sparse_dropout_with_replacement}
x!{sparse_dropout_with_replacement_cpu}
x!{test_sparse_dropout_with_replacement}
