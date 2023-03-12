#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{copy}
x!{copy_cpu_to_gpu}
x!{copy_from_cpu}
x!{copy_gpu_to_cpu}
x!{copy_on_device_like}
x!{copy_rows_to_tensor}
x!{copy_rows_to_tensor_gradient}
x!{get_copy_gradient}
x!{get_copy_rows_to_tensor_gradient}
x!{get_cpu_to_gpu_gradient}
x!{get_gpu_to_cpu_gradient}
x!{test_copy}
