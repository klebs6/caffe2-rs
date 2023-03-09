#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_softsign_gradient}
x!{softsign}
x!{softsign_cpu}
x!{softsign_gradient}
x!{softsign_gradient_cpu}
x!{test_softsign}
