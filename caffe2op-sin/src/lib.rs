#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_sin_gradient}
x!{sin}
x!{sin_functor}
x!{sin_gradient}
x!{sin_gradient_functor}
x!{test_sin}
