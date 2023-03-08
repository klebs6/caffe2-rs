#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{reverse_packed_segs}
x!{run_on_device}
x!{get_gradient}
