#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{atan}
x!{atan_gradient}
x!{get_gradient}
