#![feature(associated_type_defaults)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{activation}
x!{apply_layer_stack}
x!{cat}
x!{cell_params}
x!{chunk}
x!{full_bidirectional_lstm_layer}
x!{full_lstm_layer}
x!{gather_params}
x!{get_gradient}
x!{inference_lstm_op}
x!{layer_output}
x!{linear}
x!{lstm_cell}
x!{lstm_impl}
x!{lstm_unit}
x!{lstm_unit_gradient}
x!{lstm_unit_gradient_op}
x!{lstm_unit_op}
x!{matmul}
x!{muladd}
x!{pair_vec}
x!{stack}
x!{traits}
x!{transpose}
x!{unbind}
