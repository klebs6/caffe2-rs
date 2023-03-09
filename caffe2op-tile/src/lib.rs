#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_tile_gradient}
x!{run_on_cpu}
x!{run_on_device}
x!{test_tile}
x!{tile}
x!{tile_gradient}
