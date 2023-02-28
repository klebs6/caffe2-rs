#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{put_ops}
x!{timer_begin}
x!{timer_end}
x!{timer_get_and_end}
x!{timer_get}
x!{timer_instance}
x!{stat_registry_create}
x!{stat_registry_update}
x!{stat_registry_export}
