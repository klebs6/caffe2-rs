#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_filler}
x!{op_given_tensor_byte_string_to_uint8_fill}
x!{op_given_tensor_fill}
