#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{bucketize}
x!{bucketize_cpu}
