crate::ix!();

/**
 | Like bbox_transform_upright, but works on rotated
 | boxes.
 |
 | boxes: pixel coordinates of the bounding boxes
 |     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
 |
 | deltas: bounding box translations and scales
 |     size (M, 5), format [dx; dy; dw; dh; da]
 |     dx, dy: scale-invariant translation of the center of the bounding box
 |     dw, dh: log-space scaling of the width and height of the bounding box
 |     da: delta for angle in radians
 |
 | return: pixel coordinates of the bounding boxes
 |     size (M, 5), format [ctr_x; ctr_y; width;
 |     height; angle (in degrees)]
 |
 |     const std::vector<typename Derived2::Scalar>&
 |     weights = std::vector<typename
 |     Derived2::Scalar>{1.0, 1.0, 1.0, 1.0},
 */
#[inline] pub fn bbox_transform_rotated<Derived1: HasScalarType, Derived2: HasScalarType<Scalar = f32>>(
    boxes:           &ArrayBase<Derived1>,
    deltas:          &ArrayBase<Derived2>,
    weights:         Option<&Vec<<Derived2 as HasScalarType>::Scalar>>,
    bbox_xform_clip: Option<f32>,
    angle_bound_on:  Option<bool>,
    angle_bound_lo:  Option<i32>,
    angle_bound_hi:  Option<i32>) -> EArrXXt<<Derived1 as HasScalarType>::Scalar> {

    let bbox_xform_clip: f32 = bbox_xform_clip.unwrap_or(BBOX_XFORM_CLIP_DEFAULT);
    let angle_bound_on: bool = angle_bound_on.unwrap_or(true);
    let angle_bound_lo:  i32 = angle_bound_lo.unwrap_or(-90);
    let angle_bound_hi:  i32 = angle_bound_hi.unwrap_or(90);
    let weights              = weights.unwrap_or(&vec![1.0, 1.0, 1.0, 1.0]);

    todo!();
    /*
        using T = typename Derived1::Scalar;
      using EArrXX = EArrXXt<T>;

      if (boxes.rows() == 0) {
        return EArrXX::Zero(T(0), deltas.cols());
      }

      CAFFE_ENFORCE_EQ(boxes.rows(), deltas.rows());
      CAFFE_ENFORCE_EQ(boxes.cols(), 5);
      CAFFE_ENFORCE_EQ(deltas.cols(), 5);

      const auto& ctr_x = boxes.col(0);
      const auto& ctr_y = boxes.col(1);
      const auto& widths = boxes.col(2);
      const auto& heights = boxes.col(3);
      const auto& angles = boxes.col(4);

      auto dx = deltas.col(0).template cast<T>() / weights[0];
      auto dy = deltas.col(1).template cast<T>() / weights[1];
      auto dw =
          (deltas.col(2).template cast<T>() / weights[2]).cwiseMin(bbox_xform_clip);
      auto dh =
          (deltas.col(3).template cast<T>() / weights[3]).cwiseMin(bbox_xform_clip);
      // Convert back to degrees
      auto da = deltas.col(4).template cast<T>() * 180.0 / PI;

      EArrXX pred_boxes = EArrXX::Zero(deltas.rows(), deltas.cols());
      // new ctr_x
      pred_boxes.col(0) = dx * widths + ctr_x;
      // new ctr_y
      pred_boxes.col(1) = dy * heights + ctr_y;
      // new width
      pred_boxes.col(2) = dw.exp() * widths;
      // new height
      pred_boxes.col(3) = dh.exp() * heights;
      // new angle
      pred_boxes.col(4) = da + angles;

      if (angle_bound_on) {
        // Normalize angle to be within [angle_bound_lo, angle_bound_hi].
        // Deltas are guaranteed to be <= period / 2 while computing training
        // targets by bbox_transform_inv.
        const int period = angle_bound_hi - angle_bound_lo;
        CAFFE_ENFORCE(period > 0 && period % 180 == 0);
        auto angles = pred_boxes.col(4);
        for (const auto i : c10::irange(angles.size())) {
          if (angles[i] < angle_bound_lo) {
            angles[i] += T(period);
          } else if (angles[i] > angle_bound_hi) {
            angles[i] -= T(period);
          }
        }
      }

      return pred_boxes;
    */
}
