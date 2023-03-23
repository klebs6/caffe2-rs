#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{crop_transpose_image}
x!{image_input}
x!{image_input_cpu}
x!{image_input_cuda}
x!{random_sized_cropping}
x!{register}
x!{transform_image}
x!{transform_on_gpu}
x!{types}
