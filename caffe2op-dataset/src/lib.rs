#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{append_op}
x!{atomic_append_op}
x!{check_dataset_consistency_op}
x!{collect_tensor_op}
x!{compute_offset_op}
x!{concat_tensor_vector_op}
x!{crate_tree_cursor_op}
x!{create_tensor_vector_op}
x!{deser}
x!{get_cursor_offset_op}
x!{pack_records_op}
x!{read_next_batch_op}
x!{read_random_batch_ops}
x!{register}
x!{reset_cursor_op}
x!{sort_and_shuffle_op}
x!{tensor_vector_size_op}
x!{tree_cursor}
x!{tree_iterator}
x!{tree_walker}
x!{tree_walker_field}
x!{trim_dataset_op}
x!{unpack_records_op}
