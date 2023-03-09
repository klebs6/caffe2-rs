#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{create_scope}
x!{has_scope}
x!{workspace_stack}
