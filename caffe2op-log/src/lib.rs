#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{log}
x!{get_gradient}
x!{test_log}
