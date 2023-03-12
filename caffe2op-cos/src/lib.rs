#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cos}
x!{cos_gradient}
x!{cosine_embedding_criterion}
x!{cosine_embedding_criterion_cpu}
x!{cosine_embedding_criterion_gradient}
x!{get_cos_gradient}
x!{get_cosine_embedding_criterion_gradient}
x!{register}
x!{test_cos}
