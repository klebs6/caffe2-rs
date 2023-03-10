crate::ix!();

num_inputs!{CollectAndDistributeFpnRpnProposals, (2,INT_MAX)}

num_outputs!{CollectAndDistributeFpnRpnProposals, (3,INT_MAX)}

inputs!{CollectAndDistributeFpnRpnProposals, 
    0 => ("rpn_rois_fpn2",         "RPN proposals for FPN level 2, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals."),
    1 => ("rpn_rois_fpn3",         "RPN proposals for FPN level 3, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals."),
    2 => ("rpn_rois_fpn4",         "RPN proposals for FPN level 4, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals."),
    3 => ("rpn_rois_fpn5",         "RPN proposals for FPN level 5, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals."),
    4 => ("rpn_rois_fpn6",         "RPN proposals for FPN level 6, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals."),
    5 => ("rpn_roi_probs_fpn2",    "RPN objectness probabilities for FPN level 2. See rpn_roi_probs documentation from GenerateProposals."),
    6 => ("rpn_roi_probs_fpn3",    "RPN objectness probabilities for FPN level 3. See rpn_roi_probs documentation from GenerateProposals."),
    7 => ("rpn_roi_probs_fpn4",    "RPN objectness probabilities for FPN level 4. See rpn_roi_probs documentation from GenerateProposals."),
    8 => ("rpn_roi_probs_fpn5",    "RPN objectness probabilities for FPN level 5. See rpn_roi_probs documentation from GenerateProposals."),
    9 => ("rpn_roi_probs_fpn6",    "RPN objectness probabilities for FPN level 6. See rpn_roi_probs documentation from GenerateProposals.")
}

outputs!{CollectAndDistributeFpnRpnProposals, 
    0 => ("rois",                  "Top proposals limited to rpn_post_nms_topN total, format (image_index, x1, y1, x2, y2)"),
    1 => ("rois_fpn2",             "RPN proposals for ROI level 2, format (image_index, x1, y1, x2, y2)"),
    2 => ("rois_fpn3",             "RPN proposals for ROI level 3, format (image_index, x1, y1, x2, y2)"),
    3 => ("rois_fpn4",             "RPN proposals for ROI level 4, format (image_index, x1, y1, x2, y2)"),
    4 => ("rois_fpn5",             "RPN proposals for ROI level 5, format (image_index, x1, y1, x2, y2)"),
    5 => ("rois_idx_restore",      "Permutation on the concatenation of all rois_fpni, i=min...max, such that when applied the RPN RoIs are restored to their original order in the input blobs.")
}

args!{CollectAndDistributeFpnRpnProposals, 
    0 => ("roi_canonical_scale",   "(int) ROI_CANONICAL_SCALE"),
    1 => ("roi_canonical_level",   "(int) ROI_CANONICAL_LEVEL"),
    2 => ("roi_max_level",         "(int) ROI_MAX_LEVEL"),
    3 => ("roi_min_level",         "(int) ROI_MIN_LEVEL"),
    4 => ("rpn_max_level",         "(int) RPN_MAX_LEVEL"),
    5 => ("rpn_min_level",         "(int) RPN_MIN_LEVEL"),
    6 => ("rpn_post_nms_topN",     "(int) RPN_POST_NMS_TOP_N")
}

should_not_do_gradient!{
    CollectAndDistributeFpnRpnProposals
}
