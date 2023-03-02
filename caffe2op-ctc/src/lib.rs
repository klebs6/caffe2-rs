#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{ctc_beam_search_decoder}
x!{ctc_beam_search_decoder_cpu}

x!{ctc_greedy_decoder}
x!{ctc_greedy_decoder_cpu}

x!{get_tensor_data_ptr}
