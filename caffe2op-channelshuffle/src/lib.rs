#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{channel_shuffle}
x!{channel_shuffle_cpu}
x!{channel_shuffle_gradient}
x!{channel_shuffle_gradient_cpu}
x!{get_gradient}
x!{nchw}
x!{nhcw}
