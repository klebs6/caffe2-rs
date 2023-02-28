#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_sqr}
x!{tests}
x!{gradient}
