#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{hip_common_miopen}
x!{hip_miopen_wrapper}
x!{op_hip_activation_ops_miopen}
x!{op_rnn_hip_recurrent_op_miopen}
