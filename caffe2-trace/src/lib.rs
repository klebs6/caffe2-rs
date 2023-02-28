#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_static_tracepoint_elfx86}
x!{core_static_tracepoint}
