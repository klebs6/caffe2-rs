#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{get_gradient_defs}
x!{get_merge_multi_list_feature_tensors_gradient}
x!{get_merge_multi_map_feature_tensors_gradient}
x!{get_merge_multi_scalar_feature_tensors_gradient}
x!{get_merge_single_list_feature_tensors_gradient}
x!{get_merge_single_map_feature_tensors_gradient}
x!{get_merge_single_scalar_feature_tensors}
x!{merge_dense_feature_tensors}
x!{merge_multi_list_feature_tensors}
x!{merge_multi_list_or_map_feature_tensors_gradient}
x!{merge_multi_map_feature_tensors}
x!{merge_multi_map_feature_tensors_gradient}
x!{merge_multi_scalar_feature_tensors}
x!{merge_multi_scalar_feature_tensors_gradient}
x!{merge_single_list_feature_tensors}
x!{merge_single_list_feature_tensors_gradient}
x!{merge_single_list_or_map_feature_tensors_gradient}
x!{merge_single_map_feature_tensors}
x!{merge_single_scalar_feature_tensors}
x!{merge_single_scalar_feature_tensors_gradient}
x!{register}
