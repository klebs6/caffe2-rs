#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{constant_fill}
x!{diagonal_fill}
x!{filler}
x!{gaussian_fill}
x!{given_tensor_bool_fill}
x!{given_tensor_bytestring_to_u8fill}
x!{given_tensor_double_fill}
x!{given_tensor_fill}
x!{given_tensor_i16_fill}
x!{given_tensor_i64_fill}
x!{given_tensor_int_fill}
x!{given_tensor_string_fill}
x!{inference}
x!{lengths_range_fill}
x!{msra_fill}
x!{range_fill}
x!{register}
x!{test_constant_fill}
x!{test_gaussian_fill}
x!{test_given_tensor_bytestring_to_u8fill}
x!{test_given_tensor_fill}
x!{test_lengths_range_fill}
x!{test_uniform_fill}
x!{test_xavier_fill}
x!{uniform_fill}
x!{unique_uniform_fill}
x!{xavier_fill}
