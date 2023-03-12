#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{gpu_fallback}
x!{increment_by_one}
x!{test_increment}
