#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_tan_gradient}
x!{tan}
x!{tan_gradient}
