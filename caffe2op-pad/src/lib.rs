#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{pad_image}
x!{pad_image_cpu}
x!{pad_image_gradient}
x!{pad_image_gradient_cpu}
x!{pad_mode}
