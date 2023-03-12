#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{find}
x!{find_duplicate_elements}
x!{run_find}
x!{run_find_duplicate_elements}
x!{test_find_duplicate_elements}
