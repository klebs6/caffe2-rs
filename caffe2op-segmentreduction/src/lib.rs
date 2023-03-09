#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{abstract_lengths}
x!{abstract_lengths_def}
x!{abstract_lengths_gradient}
x!{abstract_lengths_with_main_input_and_forward_output_gradient}
x!{abstract_lengths_with_main_input_gradient}
x!{abstract_reduce_back_def}
x!{abstract_reduce_front_def}
x!{abstract_reduce_front_or_back}
x!{abstract_reduce_front_or_back_gradient}
x!{abstract_sorted_segment}
x!{abstract_sorted_segment_def}
x!{abstract_sorted_segment_gradient}
x!{abstract_sorted_segment_range}
x!{abstract_sorted_segment_range_def}
x!{abstract_sorted_segment_range_gradient}
x!{abstract_sparse_lengths_def}
x!{abstract_sparse_sorted_segment_def}
x!{abstract_sparse_unsorted_segment}
x!{abstract_unsorted_segment}
x!{abstract_unsorted_segment_def}
x!{abstract_unsorted_segment_gradient}
x!{base_input_accessor}
x!{equal}
x!{format_doc}
x!{forward}
x!{get_reduce_back_gradient}
x!{get_reduce_front_gradient}
x!{get_sorted_segment_range_gradient}
x!{inference}
x!{lengths_op}
x!{register}
x!{segment_op_get_gradient}
x!{test_segment}
