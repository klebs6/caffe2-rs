#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{pack_segments}
x!{pack_segments_config}
x!{pack_segments_cpu}
x!{pack_segments_gradient}
x!{pack_segments_run}
x!{unpack_segments}
x!{unpack_segments_config}
x!{unpack_segments_cpu}
x!{unpack_segments_gradient}
x!{unpack_segments_run}
