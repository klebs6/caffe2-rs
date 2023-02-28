#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_int8_serialization}
x!{core_serialization_test}
x!{serialization}
x!{serialize_crc_alt}
x!{serialize_crc}
x!{serialize_file_adapter}
x!{serialize_inline_container_test}
x!{serialize_inline_container}
x!{serialize_istream_adapter}
x!{serialize_read_adapter_interface}
x!{serialize_versions}
x!{tensor_deserializer}
x!{tensor_serializer}
