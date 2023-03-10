crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CollectRpnProposalsOp<Context> {
    storage:            OperatorStorage,
    context:            Context,
    rpn_max_level_:     i32, // {6};
    rpn_min_level_:     i32, // {2};
    rpn_post_nms_topN_: i32, // {2000};
}

num_inputs!{CollectRpnProposals, (2,INT_MAX)}

num_outputs!{CollectRpnProposals, 1}

inputs!{CollectRpnProposals, 
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

outputs!{CollectRpnProposals, 
    0 => ("rois",                  "Top proposals limited to rpn_post_nms_topN total, format (image_index, x1, y1, x2, y2)")
}

args!{CollectRpnProposals, 
    0 => ("rpn_max_level",         "(int) RPN_MAX_LEVEL"),
    1 => ("rpn_min_level",         "(int) RPN_MIN_LEVEL"),
    2 => ("rpn_post_nms_topN",     "(int) RPN_POST_NMS_TOP_N")
}

should_not_do_gradient!{CollectRpnProposals}

impl<Context> CollectRpnProposalsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            rpn_max_level_(
                this->template GetSingleArgument<int>("rpn_max_level", 6)),
            rpn_min_level_(
                this->template GetSingleArgument<int>("rpn_min_level", 2)),
            rpn_post_nms_topN_(
                this->template GetSingleArgument<int>("rpn_post_nms_topN", 2000)) 

        CAFFE_ENFORCE_GE(
            rpn_max_level_,
            rpn_min_level_,
            "rpn_max_level " + c10::to_string(rpn_max_level_) +
                " must be greater than or equal to rpn_min_level " +
                c10::to_string(rpn_min_level_) + ".");
        */
    }
}
