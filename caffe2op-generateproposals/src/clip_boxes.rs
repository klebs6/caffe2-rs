crate::ix!();

/**
  | Clip boxes to image boundaries.
  |
  */
#[inline] pub fn clip_boxes<Derived: HasScalarType>(
    boxes:           &ArrayBase<Derived>,
    height:          i32,
    width:           i32,
    angle_thresh:    Option<f32>,
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived as HasScalarType>::Scalar> {

    let angle_thresh:     f32 = angle_thresh.unwrap_or(1.0);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
      if (boxes.cols() == 4) {
        // Upright boxes
        return clip_boxes_upright(boxes, height, width, legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return clip_boxes_rotated(
            boxes, height, width, angle_thresh, legacy_plus_one);
      }
    */
}
