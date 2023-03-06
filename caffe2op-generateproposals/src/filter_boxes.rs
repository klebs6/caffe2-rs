crate::ix!();

#[inline] pub fn filter_boxes<Derived>(
    boxes:           &ArrayBase<Derived>,
    min_size:        f64,
    im_info:         &Array3f,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
      if (boxes.cols() == 4) {
        // Upright boxes
        return filter_boxes_upright(boxes, min_size, im_info, legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return filter_boxes_rotated(boxes, min_size, im_info);
      }
    */
}
