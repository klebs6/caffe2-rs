#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{histogram_net_observer}
x!{histogram_observer}
x!{output_column_max_histogram_net_observer}
x!{output_column_max_histogram_observer}
