#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{blob}
x!{blob_state}
x!{checkpoint_op}
x!{db_exists}
x!{format_string}
x!{get_blob_options}
x!{load_op}
x!{load_op_cpu}
x!{load_op_cuda}
x!{load_tensor_inference}
x!{register}
x!{save_op}
