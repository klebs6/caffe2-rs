#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add_padding}
x!{add_padding_cpu}
x!{gather_padding}
x!{gather_padding_cpu}
x!{get_add_padding_gradient}
x!{get_remove_padding_gradient}
x!{pad_empty_samples}
x!{pad_empty_samples_cpu}
x!{register}
x!{remove_padding}
x!{remove_padding_cpu}
x!{run_on_device}
x!{test_add_padding}
x!{test_remove_padding}
