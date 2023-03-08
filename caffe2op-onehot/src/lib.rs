#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{batch_bucket_one_hot}
x!{batch_bucket_one_hot_cpu}
x!{batch_one_hot}
x!{batch_one_hot_cpu}
x!{inference}
x!{one_hot}
x!{one_hot_cpu}
x!{register}
x!{run_on_device}
x!{segment_one_hot}
x!{test_one_hot}
