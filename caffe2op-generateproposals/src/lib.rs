#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;


x!{bbox_intersection}
x!{bbox_transform}
x!{bbox_transform_rotated}
x!{bbox_transform_upright}
x!{bbox_xyxy_to_ctrwh}
x!{clip_boxes}
x!{clip_boxes_rotated}
x!{clip_boxes_upright}
x!{compute_all_anchors}
x!{compute_sorted_anchors}
x!{compute_start_index}
x!{const_tensor_view}
x!{convex_hull_graham}
x!{cpu_add_input}
x!{filter_boxes}
x!{filter_boxes_rotated}
x!{filter_boxes_upright}
x!{generate_proposals}
x!{generate_proposals_cpu}
x!{get_sub_tensor_view}
x!{gpu_add_input}
x!{has_scalar_type}
x!{nms_cpu}
x!{nms_cpu_rotated}
x!{nms_cpu_upright}
x!{nms_cpu_with_indices}
x!{nms_gpu}
x!{polygon_area}
x!{proposals_for_one_image}
x!{rotated_rect}
x!{rotated_rect_intersection_pts}
x!{soft_nms_cpu}
x!{soft_nms_cpu_rotated}
x!{soft_nms_cpu_upright}
x!{soft_nms_cpu_with_indices}
x!{test_boxes}
x!{test_generate_proposals}
x!{test_gpu}
x!{test_nms}
x!{test_nms_gpu}
