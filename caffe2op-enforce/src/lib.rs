#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{enforce_finite}
x!{ensure_clipped}
x!{ensure_cpu_output}
