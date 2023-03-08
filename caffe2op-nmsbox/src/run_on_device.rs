crate::ix!();

impl<Context> BoxWithNMSLimitOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            score_thres_(
                this->template GetSingleArgument<float>("score_thresh", 0.05)),
            nms_thres_(this->template GetSingleArgument<float>("nms", 0.3)),
            detections_per_im_(
                this->template GetSingleArgument<int>("detections_per_im", 100)),
            soft_nms_enabled_(
                this->template GetSingleArgument<bool>("soft_nms_enabled", false)),
            soft_nms_method_str_(this->template GetSingleArgument<std::string>(
                "soft_nms_method",
                "linear")),
            soft_nms_sigma_(
                this->template GetSingleArgument<float>("soft_nms_sigma", 0.5)),
            soft_nms_min_score_thres_(this->template GetSingleArgument<float>(
                "soft_nms_min_score_thres",
                0.001)),
            rotated_(this->template GetSingleArgument<bool>("rotated", false)),
            cls_agnostic_bbox_reg_(this->template GetSingleArgument<bool>(
                "cls_agnostic_bbox_reg",
                false)),
            input_boxes_include_bg_cls_(this->template GetSingleArgument<bool>(
                "input_boxes_include_bg_cls",
                true)),
            output_classes_include_bg_cls_(this->template GetSingleArgument<bool>(
                "output_classes_include_bg_cls",
                true)),
            legacy_plus_one_(
                this->template GetSingleArgument<bool>("legacy_plus_one", true)) 

        CAFFE_ENFORCE(
            soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
            "Unexpected soft_nms_method");
        soft_nms_method_ = (soft_nms_method_str_ == "linear") ? 1 : 2;

        // When input `boxes` doesn't include background class, the score will skip
        // background class and start with foreground classes directly, and put the
        // background class in the end, i.e. score[:, 0:NUM_CLASSES-1] represents
        // foreground classes and score[:,NUM_CLASSES] represents background class.
        input_scores_fg_cls_starting_id_ = (int)input_boxes_include_bg_cls_;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() > 2) {
          return DispatchHelper<TensorTypes<int, float>>::call(this, Input(2));
        } else {
          return DoRunWithType<float>();
        }
        */
    }

    /**
      | Map a class id (starting with background
      | and then foreground) from (0, 1, ...,
      | NUM_FG_CLASSES) to it's matching value in
      | box
      */
    #[inline] pub fn get_box_cls_index(&mut self, bg_fg_cls_id: i32) -> i32 {
        
        todo!();
        /*
            if (cls_agnostic_bbox_reg_) {
          return 0;
        } else if (!input_boxes_include_bg_cls_) {
          return bg_fg_cls_id - 1;
        } else {
          return bg_fg_cls_id;
        }
        */
    }

    /**
      | Map a class id (starting with background
      | and then foreground) from (0, 1, ...,
      | NUM_FG_CLASSES) to it's matching value in
      | score
      */
    #[inline] pub fn get_score_cls_index(&mut self, bg_fg_cls_id: i32) -> i32 {
        
        todo!();
        /*
            return bg_fg_cls_id - 1 + input_scores_fg_cls_starting_id_;
        */
    }
}

