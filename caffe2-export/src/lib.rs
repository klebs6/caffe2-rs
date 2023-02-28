#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_export_c10_op_to_caffe2}
x!{core_export_caffe2_op_to_c10}
