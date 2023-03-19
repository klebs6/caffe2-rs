#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{base_context}
x!{copy_bytes}
x!{cpu_context}
x!{cuda_context}
x!{cudnn_state}
x!{cudnn_workspace}
x!{cudnn_wrapper}
x!{rand}
x!{test_context}
x!{test_gpu_context}
x!{thread_local_cuda_objects}
