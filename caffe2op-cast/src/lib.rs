#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cast}
x!{cast_cpu}
x!{cast_gradient}
x!{cast_helper}
x!{test_cast}
