#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{entropy}
x!{get_gradient}
x!{jsd_cpu}
x!{jsd_gradient}
x!{jsd_gradient_cpu}
x!{jsd}
