#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{config}
x!{get_gradient}
x!{negative}
x!{negate_gradient}
x!{test_negative}
