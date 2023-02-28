#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_flexible_top_k}

x!{op_top_k}
