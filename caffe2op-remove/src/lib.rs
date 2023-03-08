#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{remove_data_blocks}
x!{run_on_device}
