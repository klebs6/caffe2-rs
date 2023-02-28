#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_event_cpu}
x!{core_event_gpu_test}
x!{core_event_gpu}
x!{core_event_test}
x!{core_event}
