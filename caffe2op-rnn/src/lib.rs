#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{op_rnn_recurrent_network_blob_fetcher}
x!{op_rnn_recurrent_network_executor_gpu}
x!{op_rnn_recurrent_network_executor_incl}
x!{op_rnn_recurrent_network_executor}
x!{op_rnn_recurrent_network}
x!{op_rnn_recurrent_op_cudnn}
