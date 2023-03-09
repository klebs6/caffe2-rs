#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{create}
x!{summarize}
x!{summarize_cpu}
