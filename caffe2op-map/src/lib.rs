#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{create_map}
x!{key_value_to_map}
x!{map_deserializer}
x!{map_serializer}
x!{map_to_key_value}
x!{register}
