#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{channel_backprop_stats}
x!{channel_backprop_stats_cpu}
