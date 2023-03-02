#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cbrt}
x!{cbrt_gradient}
