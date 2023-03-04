#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{activation}
x!{get_gradient}
x!{gru_unit}
x!{gru_unit_gradient}
