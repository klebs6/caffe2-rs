#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{atomic_fetch_add_op}
x!{check_atomic_bool_op}
x!{conditional_set_atomic_bool_op}
x!{create_atomic_bool_op}
x!{create_mutex_op}
