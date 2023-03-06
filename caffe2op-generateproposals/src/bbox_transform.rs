crate::ix!();

#[inline] pub fn bbox_transform<Derived1: HasScalarType, Derived2: HasScalarType<Scalar = f32>>(
    boxes:           &ArrayBase<Derived1>,
    deltas:          &ArrayBase<Derived2>,
    weights:         Option<&Vec<<Derived2 as HasScalarType>::Scalar>>,
    bbox_xform_clip: Option<f32>,
    legacy_plus_one: Option<bool>,
    angle_bound_on:  Option<bool>,
    angle_bound_lo:  Option<i32>,
    angle_bound_hi:  Option<i32>) -> EArrXXt<<Derived1 as HasScalarType>::Scalar> {

    let bbox_xform_clip:  f32 = bbox_xform_clip.unwrap_or(BBOX_XFORM_CLIP_DEFAULT);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);
    let angle_bound_on:  bool = angle_bound_on.unwrap_or(true);
    let angle_bound_lo:   i32 = angle_bound_lo.unwrap_or(-90);
    let angle_bound_hi:   i32 = angle_bound_hi.unwrap_or(90);
    let weights               = weights.unwrap_or(&vec![1.0, 1.0, 1.0, 1.0]);

    todo!();
    /*
        CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
      if (boxes.cols() == 4) {
        // Upright boxes
        return bbox_transform_upright(
            boxes, deltas, weights, bbox_xform_clip, legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return bbox_transform_rotated(
            boxes,
            deltas,
            weights,
            bbox_xform_clip,
            angle_bound_on,
            angle_bound_lo,
            angle_bound_hi);
      }
    */
}
