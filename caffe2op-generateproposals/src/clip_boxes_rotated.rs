crate::ix!();

/**
 | Similar to clip_boxes_upright but handles rotated
 | boxes with angle info.
 |
 | boxes: size (M, 5), format [ctr_x; ctr_y; width;
 | height; angle (in degrees)]
 |
 | Clipping is only performed for boxes that are
 | almost upright (within a given `angle_thresh`
 | tolerance) to maintain backward compatibility for
 | non-rotated boxes.
 |
 | We don't clip rotated boxes due to a couple of
 | reasons:
 |
 | (1) There are potentially multiple ways to clip
 |     a rotated box to make it fit within the image.
 |
 | (2) It's tricky to make the entire rectangular box
 |     fit within the image and still be able to not
 |     leave out pixels of interest.
 |
 | Therefore, we rely on upstream ops like
 | RoIAlignRotated safely handling this.
 */
#[inline] pub fn clip_boxes_rotated<Derived: HasScalarType>(
    boxes:           &ArrayBase<Derived>,
    height:          i32,
    width:           i32,
    angle_thresh:    Option<f32>,
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived as HasScalarType>::Scalar> {

    let angle_thresh:     f32 = angle_thresh.unwrap_or(1.0);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(boxes.cols() == 5);

      const auto& angles = boxes.col(4);

      // Filter boxes that are upright (with a tolerance of angle_thresh)
      EArrXXt<typename Derived::Scalar> upright_boxes;
      const auto& indices = GetArrayIndices(angles.abs() <= angle_thresh);
      GetSubArrayRows(boxes, AsEArrXt(indices), &upright_boxes);

      // Convert to [x1, y1, x2, y2] format and clip them
      const auto& upright_boxes_xyxy =
          bbox_ctrwh_to_xyxy(upright_boxes.leftCols(4), legacy_plus_one);
      const auto& clipped_upright_boxes_xyxy =
          clip_boxes_upright(upright_boxes_xyxy, height, width, legacy_plus_one);

      // Convert back to [x_ctr, y_ctr, w, h, angle] and update upright boxes
      upright_boxes.block(0, 0, upright_boxes.rows(), 4) =
          bbox_xyxy_to_ctrwh(clipped_upright_boxes_xyxy, legacy_plus_one);

      EArrXXt<typename Derived::Scalar> ret(boxes.rows(), boxes.cols());
      ret = boxes;
      for (int i = 0; i < upright_boxes.rows(); ++i) {
        ret.row(indices[i]) = upright_boxes.row(i);
      }
      return ret;
    */
}
