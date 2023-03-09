#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{weighted_multi_sampling}
x!{run_on_device}
x!{inference}
