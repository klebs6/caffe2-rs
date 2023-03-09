#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient_defs}
x!{run_gradient_on_device}
x!{run_on_device}
x!{slice}
x!{slice_gradient}
x!{slice_impl}
x!{test_slice_op}
