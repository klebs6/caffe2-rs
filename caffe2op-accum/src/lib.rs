#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{accumulate}
x!{run_on_device}
