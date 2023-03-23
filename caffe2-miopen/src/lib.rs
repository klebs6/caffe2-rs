#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{check}
x!{enforce}
x!{error_string}
x!{miopen_activation_gradient}
x!{miopen_activation_op}
x!{miopen_activation_op_base}
x!{miopen_state}
x!{miopen_workspace}
x!{miopen_wrapper}
x!{recurrent_base_op}
x!{recurrent_gradient_op}
x!{recurrent_op}
x!{recurrent_param_access}
x!{synced_miopen_state}
x!{tensor_desc_wrapper}
x!{tensor_descriptors}
x!{type_wrapper}
x!{version}
