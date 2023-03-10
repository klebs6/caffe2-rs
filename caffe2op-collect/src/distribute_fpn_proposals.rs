crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DistributeFpnProposalsOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    roi_canonical_scale: i32, // {224};
    roi_canonical_level: i32, // {4};
    roi_max_level:       i32, // {5};
    roi_min_level:       i32, // {2};

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one: bool, // {true};
}

num_inputs!{DistributeFpnProposals, 1}

num_outputs!{DistributeFpnProposals, (2,INT_MAX)}

should_not_do_gradient!{DistributeFpnProposals}

inputs!{DistributeFpnProposals, 
    0 => ("rois", "Top proposals limited to rpn_post_nms_topN total, format (image_index, x1, y1, x2, y2)")
}

outputs!{DistributeFpnProposals, 
    0 => ("rois_fpn2", "RPN proposals for ROI level 2, format (image_index, x1, y1, x2, y2)"),
    1 => ("rois_fpn3", "RPN proposals for ROI level 3, format (image_index, x1, y1, x2, y2)"),
    2 => ("rois_fpn4", "RPN proposals for ROI level 4, format (image_index, x1, y1, x2, y2)"),
    3 => ("rois_fpn5", "RPN proposals for ROI level 5, format (image_index, x1, y1, x2, y2)"),
    4 => ("rois_idx_restore", "Permutation on the concatenation of all rois_fpni, i=min...max, such that when applied the RPN RoIs are restored to their original order in the input blobs.")
}

args!{DistributeFpnProposals, 
    0 => ("roi_canonical_scale", "(int) ROI_CANONICAL_SCALE"),
    1 => ("roi_canonical_level", "(int) ROI_CANONICAL_LEVEL"),
    2 => ("roi_max_level", "(int) ROI_MAX_LEVEL"),
    3 => ("roi_min_level", "(int) ROI_MIN_LEVEL")
}

impl<Context> DistributeFpnProposalsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            roi_canonical_scale_(
                this->template GetSingleArgument<int>("roi_canonical_scale", 224)),
            roi_canonical_level_(
                this->template GetSingleArgument<int>("roi_canonical_level", 4)),
            roi_max_level_(
                this->template GetSingleArgument<int>("roi_max_level", 5)),
            roi_min_level_(
                this->template GetSingleArgument<int>("roi_min_level", 2)),
            legacy_plus_one_(
                this->template GetSingleArgument<bool>("legacy_plus_one", true)) 

        CAFFE_ENFORCE_GE(
            roi_max_level_,
            roi_min_level_,
            "roi_max_level " + c10::to_string(roi_max_level_) +
                " must be greater than or equal to roi_min_level " +
                c10::to_string(roi_min_level_) + ".");
        */
    }
}
