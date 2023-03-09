#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{unsafe_coalesce}
x!{run_on_device}
