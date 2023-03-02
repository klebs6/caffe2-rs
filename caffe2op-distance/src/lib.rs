#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cosine_similarity}
x!{cosine_similarity_cpu}
x!{cosine_similarity_gradient}
x!{cosine_similarity_gradient_cpu}
x!{dot_product}
x!{dot_product_cpu}
x!{dot_product_gradient}
x!{dot_product_gradient_cpu}
x!{dot_product_inference}
x!{dot_product_with_padding}
x!{dot_product_with_padding_cpu}
x!{dot_product_with_padding_gradient}
x!{l1_distance}
x!{l1_distance_cpu}
x!{l1_distance_gradient}
x!{l1_distance_gradient_cpu}
x!{squared_l2_distance}
x!{squared_l2_distance_cpu}
x!{squared_l2_distance_gradient}
x!{test_cosine_similarity}
x!{test_dot_product}
x!{test_l1_distance}

