#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{acos}
x!{acos_gradient}
x!{get_gradient}
