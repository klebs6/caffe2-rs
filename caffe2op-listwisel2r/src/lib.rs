#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{argsort}
x!{get_gradient}
x!{lambda_rank}
x!{lambda_rank_cpu}
x!{lambda_rank_gradient}
x!{lambda_rank_gradient_cpu}
x!{macros}
