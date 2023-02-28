#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_context_base}
x!{core_context_gpu_test}
x!{core_context_gpu}
x!{core_context_test}
x!{core_context}
x!{core_cudnn_wrappers}
