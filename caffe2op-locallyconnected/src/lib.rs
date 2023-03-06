#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{locally_connected}
x!{locally_connected_gradient}
x!{register}
x!{set_column_buffer_shape}
x!{set_ybuffer_shape}
x!{shape_params}
