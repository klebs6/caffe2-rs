#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{argsort}
x!{boxes_area}
x!{collect_and_distribute_fpn_rpn_proposals}
x!{collect_and_distribute_fpn_rpn_proposals_cpu}
x!{collect_rpn_proposals}
x!{collect_rpn_proposals_cpu}
x!{config}
x!{create}
x!{distribute_fpn_proposals}
x!{distribute_fpn_proposals_cpu}
x!{map_ro_is_to_fpn_levels}
x!{register}
x!{rows_where_roi_level_equals}
x!{sort_and_limit_rois_by_scores}
