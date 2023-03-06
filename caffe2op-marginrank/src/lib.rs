#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{margin_ranking_criterion}
x!{margin_ranking_criterion_cpu}
x!{margin_ranking_criterion_gradient}
x!{margin_ranking_criterion_gradient_cpu}
