#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{queue_blobs_queue_db}
x!{queue_blobs_queue}
x!{queue_queue_ops}
x!{queue_rebatching_queue_ops}
x!{queue_rebatching_queue}
