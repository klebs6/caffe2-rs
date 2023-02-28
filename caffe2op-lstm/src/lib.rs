#![feature(associated_type_defaults)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_inference_lstm}
x!{op_lstm_unit}
x!{op_lstm_utils}
