#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{asin}
x!{asin_gradient}
x!{get_gradient}
