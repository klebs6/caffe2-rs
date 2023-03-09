#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{scale}
x!{scale_blobs}
x!{scale_cuda}
x!{run_on_device}
