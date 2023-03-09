#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{run_on_device}
x!{sinusoid_position_encoding}
