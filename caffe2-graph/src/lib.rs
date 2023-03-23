#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add_op}
x!{compare_netdefs}
x!{graph}
x!{match_arguments}
x!{match_string}
x!{node}
x!{test_graph}
