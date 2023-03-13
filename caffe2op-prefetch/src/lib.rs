#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{create}
x!{finalize}
x!{prefetch}
x!{prefetch_worker}
x!{run}
