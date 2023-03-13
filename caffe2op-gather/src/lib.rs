#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{calc_output_shape}
x!{check_indexarray}
x!{do_gather}
x!{gather}
x!{gather_fused}
x!{gather_ranges_to_dense}
x!{get_gather_gradient}
x!{run_gather}
x!{run_gather_fused}
x!{run_gather_ranges_to_dense}
x!{test_gather}
x!{test_gather_ranges_to_dense}
