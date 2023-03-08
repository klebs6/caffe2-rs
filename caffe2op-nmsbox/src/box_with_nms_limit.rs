crate::ix!();

/**
  | Apply NMS to each class (except background)
  | and limit the number of returned boxes.
  | 
  | C++ implementation of function insert_box_results_with_nms_and_limit()
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BoxWithNMSLimitOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /// TEST.SCORE_THRESH
    score_thres: f32, // 0.05

    /// TEST.NMS
    nms_thres: f32, //0.3

    /// TEST.DETECTIONS_PER_IM
    detections_per_im: i32, //100

    /// TEST.SOFT_NMS.ENABLED
    soft_nms_enabled: bool, //false

    /// TEST.SOFT_NMS.METHOD
    soft_nms_method_str: String, // "linear"
    soft_nms_method: u32, //1, linear

    /// TEST.SOFT_NMS.SIGMA
    soft_nms_sigma: f32, //0.5

    /**
      | Lower-bound on updated scores to discard
      | boxes
      |
      */
    soft_nms_min_score_thres: f32, //0.001

    /**
      | Set for RRPN case to handle rotated boxes.
      | Inputs should be in format [ctr_x, ctr_y,
      | width, height, angle (in degrees)].
      |
      */
    rotated: bool, //false

    /// MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
    cls_agnostic_bbox_reg: bool, //false

    /**
      | Whether input `boxes` includes background
      | class. If true, boxes will have shape of
      | (N, (num_fg_class+1) * 4or5), otherwise
      | (N, num_fg_class * 4or5)
      */
    input_boxes_include_bg_cls: bool, //true

    /**
      | Whether output `classes` includes
      | background class. If true, index 0 will
      | represent background, and valid outputs
      | start from 1.
      */
    output_classes_include_bg_cls: bool, //true

    /**
      | The index where foreground starts in
      | scoures. Eg. if 0 represents background
      | class then foreground class starts
      | with 1.
      |
      */
    input_scores_fg_cls_starting_id: i32, //1

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one: bool, //true
}
