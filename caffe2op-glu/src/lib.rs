#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{glu}
x!{glu_cpu}
x!{run_on_device}
x!{sigmoid}
