#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{accumulate_input_gradient}
x!{add_apply_link_ops}
x!{components}
x!{create_rnnexecutor}
x!{cuda_recurrent_network_executor}
x!{extract_links}
x!{extract_net_def}
x!{get_gradient}
x!{get_recurrent_mapping}
x!{get_recurrent_network_gradient}
x!{initialize_recurrent_input}
x!{offset}
x!{op_task}
x!{prepend_ops}
x!{recurrent_access_op}
x!{recurrent_base_op}
x!{recurrent_gradient_op}
x!{recurrent_network}
x!{recurrent_network_blob_fetcher}
x!{recurrent_network_executor}
x!{recurrent_network_gradient}
x!{recurrent_op}
x!{recurrent_param_access}
x!{register}
x!{repeat}
x!{rnn_apply_link}
x!{rnn_net_operator}
x!{tensor_descriptors}
x!{threaded_recurrent_network_executor}
x!{timestep}
