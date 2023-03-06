#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient}
x!{integral_image}
x!{integral_image_cpu}
x!{integral_image_gradient}
x!{integral_image_gradient_cpu}
