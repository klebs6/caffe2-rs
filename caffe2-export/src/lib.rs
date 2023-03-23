#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{call_caffe_op}
x!{create_opertor_wrapper}
x!{export_op}
x!{make_function_schema}
x!{opertor_wrapper}
