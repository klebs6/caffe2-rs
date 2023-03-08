#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{local_scope}
x!{onnx_while}
x!{run_on_device}
