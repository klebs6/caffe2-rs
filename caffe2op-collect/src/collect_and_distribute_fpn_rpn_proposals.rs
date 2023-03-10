crate::ix!();

/**
  | Merge RPN proposals generated at multiple
  | FPN levels and then distribute those
  | proposals to their appropriate FPN
  | levels for Faster RCNN.
  | 
  | An anchor at one FPN level may predict
  | an RoI that will map to another level,
  | hence the need to redistribute the proposals.
  | 
  | Only inference is supported. To train,
  | please use the original Python operator
  | in Detectron.
  | 
  | Inputs and outputs are examples only;
  | if min/max levels change, the number
  | of inputs and outputs, as well as their
  | level numbering, will change.
  | 
  | C++ implementation of
  | 
  | CollectAndDistributeFpnRpnProposalsOp
  | Merge RPN proposals generated at multiple
  | FPN levels and then distribute those
  | proposals to their appropriate FPN
  | levels for Faster RCNN. An anchor at
  | one FPN level may predict an RoI that
  | will map to another level, hence the
  | need to redistribute the proposals.
  | 
  | Reference: facebookresearch/Detectron/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CollectAndDistributeFpnRpnProposalsOp<Context> {
    storage:                OperatorStorage,
    context:                Context,

    /// ROI_CANONICAL_SCALE
    roi_canonical_scale:    i32, // {224};

    /// ROI_CANONICAL_LEVEL
    roi_canonical_level:    i32, // {4};

    /// ROI_MAX_LEVEL
    roi_max_level:          i32, // {5};

    /// ROI_MIN_LEVEL
    roi_min_level:          i32, // {2};

    /// RPN_MAX_LEVEL
    rpn_max_level:          i32, // {6};

    /// RPN_MIN_LEVEL
    rpn_min_level:          i32, // {2};

    /// RPN_POST_NMS_TOP_N
    rpn_post_nms_top_n:     i32, // {2000};

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one:        bool, // {true};
}
