crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext
};

/**
  | Transform proposal bounding boxes
  | to target bounding box using bounding
  | box regression deltas.
  |
  */
pub struct BBoxTransformOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

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

register_cpu_operator!{BBoxTransform, BBoxTransformOp<f32, CPUContext>}

should_not_do_gradient!{BBoxTransform}

impl BBoxTransformOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& roi_in = Input(0);
      const auto& delta_in = Input(1);
      const auto& iminfo_in = Input(2);

      const int box_dim = rotated_ ? 5 : 4;
      const int N = roi_in.dim32(0);
      CAFFE_ENFORCE_EQ(roi_in.dim(), 2);
      CAFFE_ENFORCE(roi_in.dim32(1) == box_dim || roi_in.dim32(1) == box_dim + 1);

      CAFFE_ENFORCE_EQ(delta_in.dim(), 2);
      CAFFE_ENFORCE_EQ(delta_in.dim32(0), N);
      CAFFE_ENFORCE_EQ(delta_in.dim32(1) % box_dim, 0);
      const int num_classes = delta_in.dim32(1) / box_dim;

      CAFFE_ENFORCE_EQ(iminfo_in.dim(), 2);
      CAFFE_ENFORCE_EQ(iminfo_in.dim32(1), 3);
      const int batch_size = iminfo_in.dim32(0);

      DCHECK_EQ(weights_.size(), 4);

      Eigen::Map<const ERArrXXf> boxes0(
          roi_in.data<float>(), roi_in.dim32(0), roi_in.dim32(1));
      Eigen::Map<const ERArrXXf> deltas0(
          delta_in.data<float>(), delta_in.dim32(0), delta_in.dim32(1));

      // Count the number of RoIs per batch
      vector<int> num_rois_per_batch(batch_size, 0);
      if (roi_in.dim32(1) == box_dim) {
        CAFFE_ENFORCE_EQ(batch_size, 1);
        num_rois_per_batch[0] = N;
      } else {
        const auto& roi_batch_ids = boxes0.col(0);
        for (int i = 0; i < roi_batch_ids.size(); ++i) {
          const int roi_batch_id = roi_batch_ids(i);
          CAFFE_ENFORCE_LT(roi_batch_id, batch_size);
          num_rois_per_batch[roi_batch_id]++;
        }
      }

      CAFFE_ENFORCE_EQ(iminfo_in.sizes(), (at::IntArrayRef{batch_size, 3}));
      Eigen::Map<const ERArrXXf> iminfo(
          iminfo_in.data<float>(), iminfo_in.size(0), iminfo_in.size(1));

      auto* box_out = Output(0, delta_in.sizes(), at::dtype<float>());
      Eigen::Map<ERArrXXf> new_boxes(
          box_out->template mutable_data<float>(),
          box_out->dim32(0),
          box_out->dim32(1));

      // We assume roi_in and delta_in over multiple batches are grouped
      // together in increasing order as generated by GenerateProposalsOp
      int offset = 0;
      for (int i = 0; i < batch_size; ++i) {
        const int num_rois = num_rois_per_batch[i];
        const auto& cur_iminfo = iminfo.row(i);
        const float scale_before = cur_iminfo(2);
        const float scale_after = apply_scale_ ? cur_iminfo(2) : 1.0;
        int img_h = int(cur_iminfo(0) / scale_before + 0.5);
        int img_w = int(cur_iminfo(1) / scale_before + 0.5);

        EArrXXf cur_boxes =
            boxes0.rightCols(box_dim).block(offset, 0, num_rois, box_dim);
        // Do not apply scale for angle in rotated boxes
        cur_boxes.leftCols(4) /= scale_before;
        for (int k = 0; k < num_classes; k++) {
          const auto& cur_deltas =
              deltas0.block(offset, k * box_dim, num_rois, box_dim);
          const auto& trans_boxes = utils::bbox_transform(
              cur_boxes,
              cur_deltas,
              weights_,
              utils::BBOX_XFORM_CLIP_DEFAULT,
              legacy_plus_one_,
              angle_bound_on_,
              angle_bound_lo_,
              angle_bound_hi_);
          EArrXXf clip_boxes = utils::clip_boxes(
              trans_boxes, img_h, img_w, clip_angle_thresh_, legacy_plus_one_);
          // Do not apply scale for angle in rotated boxes
          clip_boxes.leftCols(4) *= scale_after;
          new_boxes.block(offset, k * box_dim, num_rois, box_dim) = clip_boxes;
        }

        offset += num_rois;
      }

      if (OutputSize() > 1) {
        auto* roi_batch_splits = Output(1, {batch_size}, at::dtype<float>());
        Eigen::Map<EArrXf> roi_batch_splits_map(
            roi_batch_splits->template mutable_data<float>(), batch_size);
        roi_batch_splits_map =
            Eigen::Map<const EArrXi>(num_rois_per_batch.data(), batch_size)
                .cast<float>();
      }

      return true;
        */
    }
}

pub type BBoxTransformOpFloatCPU = BBoxTransformOp<f32, CPUContext>;

