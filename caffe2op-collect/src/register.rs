crate::ix!();

register_cpu_operator!{CollectAndDistributeFpnRpnProposals, CollectAndDistributeFpnRpnProposalsOp<CPUContext>}
register_cpu_operator!{CollectRpnProposals,                 CollectRpnProposalsOp<CPUContext>}
register_cpu_operator!{DistributeFpnProposals,              DistributeFpnProposalsOp<CPUContext>}

export_caffe2_op_to_c10_cpu!{CollectAndDistributeFpnRpnProposals,
    "_caffe2::CollectAndDistributeFpnRpnProposals(
        Tensor[] input_list, 
        int roi_canonical_scale, 
        int roi_canonical_level, 
        int roi_max_level, 
        int roi_min_level, 
        int rpn_max_level, 
        int rpn_min_level, 
        int rpn_post_nms_topN, 
        bool legacy_plus_one) 
        -> (
            Tensor rois, 
            Tensor rois_fpn2, 
            Tensor rois_fpn3, 
            Tensor rois_fpn4, 
            Tensor rois_fpn5, 
            Tensor rois_idx_restore_int32)",
    CollectAndDistributeFpnRpnProposalsOp<CPUContext>}

export_caffe2_op_to_c10_cpu!{CollectRpnProposals,
    "_caffe2::CollectRpnProposals(
        Tensor[] input_list, 
        int rpn_max_level, 
        int rpn_min_level, 
        int rpn_post_nms_topN) -> (Tensor rois)",
    CollectRpnProposalsOp<CPUContext>}

export_caffe2_op_to_c10_cpu!{DistributeFpnProposals,
    "_caffe2::DistributeFpnProposals(
        Tensor rois, 
        int roi_canonical_scale, 
        int roi_canonical_level, 
        int roi_max_level, 
        int roi_min_level, 
        bool legacy_plus_one) 
        -> (
        Tensor rois_fpn2, 
        Tensor rois_fpn3, 
        Tensor rois_fpn4, 
        Tensor rois_fpn5, 
        Tensor rois_idx_restore_int32)",
        DistributeFpnProposalsOp<CPUContext>}
