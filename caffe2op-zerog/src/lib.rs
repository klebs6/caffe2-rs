#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{zero_gradient}
x!{get_gradient}
