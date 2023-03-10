crate::ix!();

impl<Context> CollectAndDistributeFpnRpnProposalsOp<Context> {
    
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
            rpn_max_level_(
                this->template GetSingleArgument<int>("rpn_max_level", 6)),
            rpn_min_level_(
                this->template GetSingleArgument<int>("rpn_min_level", 2)),
            rpn_post_nms_topN_(
                this->template GetSingleArgument<int>("rpn_post_nms_topN", 2000)),
            legacy_plus_one_(
                this->template GetSingleArgument<bool>("legacy_plus_one", true)) 

        CAFFE_ENFORCE_GE(
            roi_max_level_,
            roi_min_level_,
            "roi_max_level " + c10::to_string(roi_max_level_) +
                " must be greater than or equal to roi_min_level " +
                c10::to_string(roi_min_level_) + ".");
        CAFFE_ENFORCE_GE(
            rpn_max_level_,
            rpn_min_level_,
            "rpn_max_level " + c10::to_string(rpn_max_level_) +
                " must be greater than or equal to rpn_min_level " +
                c10::to_string(rpn_min_level_) + ".");
        */
    }
}
