#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{load_module}
x!{module_schema}
x!{test_module}
