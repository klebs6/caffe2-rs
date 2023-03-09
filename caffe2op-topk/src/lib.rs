#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{flexible_topk}
x!{flexible_topk_gradient}
x!{get_flexible_topk_gradient}
x!{get_topk}
x!{get_topk_gradient}
x!{run_flexible_topk}
x!{run_top_k}
x!{set_topk}
x!{test_top_k}
x!{topk}
x!{topk_gradient}
x!{value_cmp}
