#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{config}
x!{sparse_lp_regularizer}
x!{sparse_lp_regularizer_cpu}
