crate::ix!();

/**
  | Clip boxes to image boundaries
  |
  | boxes: pixel coordinates of bounding box, size
  | (M * 4)
  */
#[inline] pub fn clip_boxes_upright<Derived: HasScalarType>(
    boxes:           &ArrayBase<Derived>,
    height:          i32,
    width:           i32,
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived as HasScalarType>::Scalar> {

    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();

    /*
        CAFFE_ENFORCE(boxes.cols() == 4);

      EArrXXt<typename Derived::Scalar> ret(boxes.rows(), boxes.cols());

      // x1 >= 0 && x1 < width
      ret.col(0) = boxes.col(0).cwiseMin(width - int(legacy_plus_one)).cwiseMax(0);
      // y1 >= 0 && y1 < height
      ret.col(1) = boxes.col(1).cwiseMin(height - int(legacy_plus_one)).cwiseMax(0);
      // x2 >= 0 && x2 < width
      ret.col(2) = boxes.col(2).cwiseMin(width - int(legacy_plus_one)).cwiseMax(0);
      // y2 >= 0 && y2 < height
      ret.col(3) = boxes.col(3).cwiseMin(height - int(legacy_plus_one)).cwiseMax(0);

      return ret;
    */
}


