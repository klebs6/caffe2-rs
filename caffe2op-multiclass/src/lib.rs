#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{multi_class_accuracy}
x!{multi_class_accuracy_cpu}
