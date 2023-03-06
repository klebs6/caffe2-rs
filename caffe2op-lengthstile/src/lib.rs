#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{lengths_tile}
x!{lengths_tile_cpu}
x!{test_lengths_tile}
