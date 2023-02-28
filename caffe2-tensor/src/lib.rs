#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_tensor_int8}
x!{core_tensor}
x!{filler}
x!{util_smart_tensor_printer_test}
x!{util_smart_tensor_printer}
