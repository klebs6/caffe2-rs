#![feature(adt_const_params)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cpu_event_wrapper}
x!{cuda_event_create}
x!{cuda_event_error}
x!{cuda_event_finish}
x!{cuda_event_finished}
x!{cuda_event_query}
x!{cuda_event_record}
x!{cuda_event_register}
x!{cuda_event_reset}
x!{cuda_event_wait}
x!{cuda_event_wrapper}
x!{event}
x!{event_create}
x!{event_error}
x!{event_finish}
x!{event_hooks}
x!{event_query}
x!{event_record}
x!{event_register}
x!{event_registrar}
x!{event_reset}
x!{event_set_callback}
x!{event_status}
x!{event_wait}
x!{test_event_cpu}
x!{test_event_gpu}
