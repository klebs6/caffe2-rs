#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{rowwise_max_and_arg}
x!{swap_best_path}
x!{viterbi_path}
