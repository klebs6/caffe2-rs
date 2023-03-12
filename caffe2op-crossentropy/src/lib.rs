#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cross_entropy}
x!{cross_entropy_cpu}
x!{cross_entropy_gradient}
x!{cross_entropy_gradient_cpu}
x!{get_cross_entropy_gradient_cpu}
x!{get_label_cross_entropy_gradient}
x!{get_make_two_class_gradient}
x!{get_sigmoid_cross_entropy_with_logits_gradient}
x!{get_weighted_sigmoid_cross_entropy_with_logits_gradient}
x!{label_cross_entropy}
x!{label_cross_entropy_cpu}
x!{label_cross_entropy_gradient}
x!{label_cross_entropy_gradient_cpu}
x!{make_two_class}
x!{make_two_class_cpu}
x!{make_two_class_gradient}
x!{register}
x!{sigmoid}
x!{sigmoid_cross_entropy_with_logits}
x!{sigmoid_cross_entropy_with_logits_cpu}
x!{sigmoid_cross_entropy_with_logits_gradient}
x!{sigmoid_cross_entropy_with_logits_gradient_cpu}
x!{test_cross_entropy}
x!{test_label_cross_entropy}
x!{weighted_sigmoid_cross_entropy_with_logits}
x!{weighted_sigmoid_cross_entropy_with_logits_cpu}
x!{weighted_sigmoid_cross_entropy_with_logits_gradient}
x!{weighted_sigmoid_cross_entropy_with_logits_gradient_cpu}
