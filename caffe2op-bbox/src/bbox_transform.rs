crate::ix!();

/**
  | Transform proposal bounding boxes
  | to target bounding box using bounding
  | box regression deltas.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BBoxTransformOp<T, Context> {


    storage: OperatorStorage,
    context: Context,

    /**
      | weights [wx, wy, ww, wh] to apply to the
      | regression target
      |
      */
    weights_: Vec<T>,

    /**
      | Transform the boxes to the scaled image
      |   space after applying the bbox deltas.
      |
      | Set to false to match the detectron code,
      |   set to true for the keypoint model and
      |   for backward compatibility
      */
    apply_scale_: bool, //{true};

    /**
      | Set for RRPN case to handle rotated boxes.
      |
      | Inputs should be in format [ctr_x, ctr_y,
      | width, height, angle (in degrees)].
      */
    rotated_: bool, //{false};

    /**
      | If set, for rotated boxes in RRPN, output
      | angles are normalized to be within [angle_bound_lo,
      | angle_bound_hi].
      |
      */
    angle_bound_on_: bool, //{true};

    angle_bound_lo_: i32, //{-90};

    angle_bound_hi_: i32, //{90};

    /**
      | For RRPN, clip almost horizontal boxes
      | within this threshold of tolerance for
      | backward compatibility.
      |
      | Set to negative value for no clipping.
      */
    clip_angle_thresh_: f32, //{1.0};

    /**
      | The infamous "+ 1" for box width and height
      | dating back to the DPM days
      |
      */
    legacy_plus_one_: bool, //{true};
}

// Input: box, delta Output: box
num_inputs!{BBoxTransform, 3}

num_outputs!{BBoxTransform, (1,2)}

inputs!{BBoxTransform, 
    0 => ("rois", 
        "Bounding box proposals in pixel coordinates, 
            Size (M, 4), format [x1, y1, x2, y2], or
            Size (M, 5), format [batch_index, x1, y1, x2, y2]. 
            If proposals from multiple images in a batch are present, they 
            should be grouped sequentially and in incremental order.
            For rotated boxes, this would have an additional angle (in degrees) 
            in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle]."
    ),
    1 => ("deltas", 
        "bounding box translations and scales, size (M, 4*K), format [dx, dy, dw, dh], K = # classes.  For rotated boxes, size (M, 5*K, format [dx, dy, dw, dh, da]."
    ),
    2 => ("im_info", 
        "Image dimensions, size (batch_size, 3), format [img_height, img_width, img_scale]")
}

outputs!{BBoxTransform, 
    0 => ("box_out",          "Pixel coordinates of the transformed bounding boxes, Size (M, 4*K), format [x1, y1, x2, y2].  For rotated boxes, size (M, 5*K), format [ctr_x, ctr_y, w, h, angle]."),
    1 => ("roi_batch_splits", "Tensor of shape (batch_size) with each element denoting the number of RoIs belonging to the corresponding image in batch")
}

args!{BBoxTransform, 
    0 => ("weights",           "vector<float> weights [wx, wy, ww, wh] for the deltas"),
    1 => ("apply_scale",       "bool (default true), transform the boxes to the scaled image space after applying the bbox deltas. Set to false to match the detectron code, set to true for keypoint models and for backward compatibility"),
    2 => ("rotated",           "bool (default false). If true, then boxes (rois and deltas) include angle info to handle rotation. The format will be [ctr_x, ctr_y, width, height, angle (in degrees)]."),
    3 => ("angle_bound_on",    "bool (default true). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    4 => ("angle_bound_lo",    "int (default -90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    5 => ("angle_bound_hi",    "int (default 90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi]."),
    6 => ("clip_angle_thresh", "float (default 1.0 degrees). For RRPN, clip almost horizontal boxes within this threshold of tolerance for backward compatibility. Set to negative value for no clipping.")
}

register_cpu_operator!{BBoxTransform, BBoxTransformOp<f32, CPUContext>}

should_not_do_gradient!{BBoxTransform}

impl<T,Context> BBoxTransformOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            weights_(this->template GetRepeatedArgument<T>(
                "weights",
                vector<T>{1.0f, 1.0f, 1.0f, 1.0f})),
            apply_scale_(
                this->template GetSingleArgument<bool>("apply_scale", true)),
            rotated_(this->template GetSingleArgument<bool>("rotated", false)),
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
        CAFFE_ENFORCE_EQ(
            weights_.size(),
            4,
            "weights size " + c10::to_string(weights_.size()) + "must be 4.");
        */
    }
}
