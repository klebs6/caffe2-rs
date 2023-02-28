#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_copy_rows_to_tensor}
x!{op_copy}
