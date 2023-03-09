#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{add_input}
x!{bilinear_interpolate_gradient}
x!{bilinear_interpolation_params}
x!{create_and_run}
x!{get_device_type}
x!{get_roi_align_gradient}
x!{get_roi_align_rotated_gradient}
x!{pre_calc}
x!{roi_align}
x!{roi_align_backward_feature}
x!{roi_align_cpu}
x!{roi_align_gradient}
x!{roi_align_gradient_cpu}
x!{roi_align_rotated}
x!{roi_align_rotated_cpu}
x!{roi_align_rotated_forward}
x!{roi_align_rotated_gradient}
x!{test_roi_align}
