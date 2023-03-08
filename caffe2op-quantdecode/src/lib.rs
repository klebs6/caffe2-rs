#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{decode}
x!{get_gradient}
x!{quant_decode}
x!{quant_decode_gradient}
x!{quant_decode_gradient_config}
x!{test_quant_decode_gradient}
