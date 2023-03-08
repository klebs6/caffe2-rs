#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{reciprocal_functor}
x!{reciprocal_functor_config}
x!{reciprocal_gradient_functor}
x!{reciprocal_gradient_functor_config}
x!{test_reciprocal}
