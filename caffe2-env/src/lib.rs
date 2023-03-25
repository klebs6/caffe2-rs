#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{declare}
x!{define}
x!{export_op}
x!{register}
