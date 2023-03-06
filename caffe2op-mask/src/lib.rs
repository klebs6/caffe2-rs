#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add_scalar_input}
x!{boolean_mask}
x!{boolean_mask_gradient}
x!{boolean_mask_gradient_cpu}
x!{boolean_mask_lengths}
x!{boolean_mask_lengths_cpu}
x!{boolean_unmask}
x!{boolean_unmask_cpu}
x!{get_boolean_mask_gradient}
x!{get_sequence_mask_gradient}
x!{lower_functor}
x!{mask_with_functor}
x!{op_boolean_unmask_ops}
x!{repeated_mask_with_functor}
x!{sequence_functor}
x!{sequence_mask}
x!{sequence_mask_cpu}
x!{test_boolean_mask}
x!{upper_functor}
x!{window_functor}
