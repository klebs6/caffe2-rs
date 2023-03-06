crate::ix!();

/**
  | C++ implementation of GenerateProposalsOp
  | 
  | Generate bounding box proposals for
  | Faster RCNN.
  | 
  | The proposals are generated for a list
  | of images based on image score 'score',
  | bounding box regression result 'deltas'
  | as well as predefined bounding box shapes
  | 'anchors'.
  | 
  | Greedy non-maximum suppression is
  | applied to generate the final bounding
  | boxes.
  | 
  | Reference: facebookresearch/Detectron/detectron/ops/generate_proposals.py
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GenerateProposalsOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /**
      | spatial_scale_ must be declared before
      | feat_stride_
      |
      */
    spatial_scale:                  f32, // default = 1.0

    feat_stride:                    f32, // default = 1.0

    /// RPN_PRE_NMS_TOP_N
    rpn_pre_nms_topn:               i32, // default = 6000

    /// RPN_POST_NMS_TOP_N
    rpn_post_nms_topn:              i32, // default = 300

    /// RPN_NMS_THRESH
    rpn_nms_thresh:                 f32, // default = 0.7

    /// RPN_MIN_SIZE
    rpn_min_size:                   f32, // default = 16

    /**
      | If set, for rotated boxes in RRPN, output
      | angles are normalized to be within [angle_bound_lo,
      | angle_bound_hi].
      |
      */
    angle_bound_on:                 bool, // default = true

    angle_bound_lo:                 i32, // default = -90
    angle_bound_hi:                 i32, // default = 90

    /**
      | For RRPN, clip almost horizontal boxes
      | within this threshold of tolerance for
      | backward compatibility. Set to negative
      | value for no clipping.
      */
    clip_angle_thresh:              f32, // default = 1.0

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one:                bool, // default = true

    /**
      | Scratch space required by the CUDA version
      | CUB buffers
      |
      | {Context::GetDeviceType()};
      */
    dev_cub_sort_buffer:            Tensor,

    /// {Context::GetDeviceType()};
    dev_cub_select_buffer:          Tensor,

    /// {Context::GetDeviceType()};
    dev_image_offset:               Tensor,

    /// {Context::GetDeviceType()};
    dev_conv_layer_indexes:         Tensor,

    /// {Context::GetDeviceType()};
    dev_sorted_conv_layer_indexes:  Tensor,

    /// {Context::GetDeviceType()};
    dev_sorted_scores:              Tensor,

    /// {Context::GetDeviceType()};
    dev_boxes:                      Tensor,

    /// {Context::GetDeviceType()};
    dev_boxes_keep_flags:           Tensor,

    /**
      | prenms proposals (raw proposals minus
      | empty boxes)
      |
      | {Context::GetDeviceType()};
      */
    dev_image_prenms_boxes:         Tensor,

    /// {Context::GetDeviceType()};
    dev_image_prenms_scores:        Tensor,

    /// {Context::GetDeviceType()};
    dev_prenms_nboxes:              Tensor,

    /// {CPU};
    host_prenms_nboxes:             Tensor,

    /// {Context::GetDeviceType()};
    dev_image_boxes_keep_list:      Tensor,

    /**
      | Tensors used by NMS
      | 
      | {Context::GetDeviceType()};
      |
      */
    dev_nms_mask:                   Tensor,

    /// {CPU};
    host_nms_mask:                  Tensor,

    /**
      | Buffer for output
      | 
      | {Context::GetDeviceType()};
      |
      */
    dev_postnms_rois:               Tensor,

    /// {Context::GetDeviceType()};
    dev_postnms_rois_probs:         Tensor,
}

/**
  | Generate bounding box proposals for
  | Faster RCNN.
  | 
  | The propoasls are generated for a list
  | of images based on image score 'score',
  | bounding box regression result 'deltas'
  | as well as predefined bounding box shapes
  | 'anchors'.
  | 
  | Greedy non-maximum suppression is
  | applied to generate the final bounding
  | boxes.
  |
  */
register_cpu_operator!{GenerateProposals, GenerateProposalsOp<CPUContext>}

num_inputs!{GenerateProposals, 4}

num_outputs!{GenerateProposals, 2}

inputs!{GenerateProposals, 
    0 => ("scores",            "Scores from conv layer, size (img_count, A, H, W)"),
    1 => ("bbox_deltas",       "Bounding box deltas from conv layer, size (img_count, 4 * A, H, W)"),
    2 => ("im_info",           "Image info, size (img_count, 3), format (height, width, scale)"),
    3 => ("anchors",           "Bounding box anchors, size (A, 4)")
}

outputs!{GenerateProposals, 
    0 => ("rois",              "Proposals, size (n x 5), format (image_index, x1, y1, x2, y2)"),
    1 => ("rois_probs",        "scores of proposals, size (n)")
}

args!{GenerateProposals, 
    0 => ("spatial_scale",     "(float) spatial scale"),
    1 => ("pre_nms_topN",      "(int) RPN_PRE_NMS_TOP_N"),
    2 => ("post_nms_topN",     "(int) RPN_POST_NMS_TOP_N"),
    3 => ("nms_thresh",        "(float) RPN_NMS_THRESH"),
    4 => ("min_size",          "(float) RPN_MIN_SIZE"),
    5 => ("angle_bound_on",    "bool (default true). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    6 => ("angle_bound_lo",    "int (default -90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    7 => ("angle_bound_hi",    "int (default 90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    8 => ("clip_angle_thresh", "float (default 1.0 degrees). For RRPN, clip almost horizontal boxes within this threshold of tolerance for backward compatibility. Set to negative value for no clipping.")
}

// For backward compatibility
register_cpu_operator!{GenerateProposalsCPP, GenerateProposalsOp<CPUContext>}

///----------------------
// For backward compatibility
num_inputs!{GenerateProposalsCPP, 4}

num_outputs!{GenerateProposalsCPP, 2}

should_not_do_gradient!{GenerateProposals}

// For backward compatibility
should_not_do_gradient!{GenerateProposalsCPP}

export_caffe2_op_to_c10_cpu!{
    GenerateProposals,
    "_caffe2::GenerateProposals(
        Tensor scores, 
        Tensor bbox_deltas, 
        Tensor im_info, 
        Tensor anchors, 
        float spatial_scale, 
        int pre_nms_topN, 
        int post_nms_topN, 
        float nms_thresh, 
        float min_size, 
        bool angle_bound_on, 
        int angle_bound_lo, 
        int angle_bound_hi, 
        float clip_angle_thresh, 
        bool legacy_plus_one) -> (Tensor output_0, 
        Tensor output_1)",
        GenerateProposalsOp<CPUContext>
}

impl<Context> GenerateProposalsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            spatial_scale_(
                this->template GetSingleArgument<float>("spatial_scale", 1.0 / 16)),
            feat_stride_(1.0 / spatial_scale_),
            rpn_pre_nms_topN_(
                this->template GetSingleArgument<int>("pre_nms_topN", 6000)),
            rpn_post_nms_topN_(
                this->template GetSingleArgument<int>("post_nms_topN", 300)),
            rpn_nms_thresh_(
                this->template GetSingleArgument<float>("nms_thresh", 0.7f)),
            rpn_min_size_(this->template GetSingleArgument<float>("min_size", 16)),
            angle_bound_on_(
                this->template GetSingleArgument<bool>("angle_bound_on", true)),
            angle_bound_lo_(
                this->template GetSingleArgument<int>("angle_bound_lo", -90)),
            angle_bound_hi_(
                this->template GetSingleArgument<int>("angle_bound_hi", 90)),
            clip_angle_thresh_(
                this->template GetSingleArgument<float>("clip_angle_thresh", 1.0)),
            legacy_plus_one_(
                this->template GetSingleArgument<bool>("legacy_plus_one", true))
        */
    }
}
