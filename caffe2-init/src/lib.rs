#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_init_denormals}
x!{core_init_intrinsics_check}
x!{core_init_omp}
x!{core_init_test}
x!{core_init}
