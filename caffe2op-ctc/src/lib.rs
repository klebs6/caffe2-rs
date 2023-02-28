#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_ctc_beam_search_decoder}
x!{op_ctc_greedy_decoder}
