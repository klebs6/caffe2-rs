#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{pack_rnnseq}
x!{run_on_device}
x!{unpack_rnnseq}
