#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sinh_gradient}
x!{sinh}
x!{sinh_gradient}
x!{test_sinh}
