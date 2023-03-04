#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{hsoftmax}
x!{hsoftmax_base}
x!{hsoftmax_cpu}
x!{hsoftmax_gradient}
x!{hsoftmax_gradient_cpu}
x!{hsoftmax_search}
x!{hsoftmax_search_cpu}
x!{huffman_tree_hierarchy}
x!{huffman_tree_hierarchy_node}
