#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{box_with_nms_limit}
x!{box_with_nms_limit_cpu}
x!{config}
x!{run_on_device}
