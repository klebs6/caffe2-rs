crate::ix!();

/**
  | Similar to filter_boxes_upright but
  | works for rotated boxes.
  | 
  | boxes: pixel coordinates of the bounding
  | boxes size (M, 5), format [ctr_x; ctr_y;
  | width; height; angle (in degrees)]
  | 
  | im_info: [height, width, img_scale]
  | 
  | return: row indices for 'boxes'
  |
  */
#[inline] pub fn filter_boxes_rotated<Derived>(
    boxes:    &ArrayBase<Derived>,
    min_size: f64,
    im_info:  &Array3f) -> Vec<i32> {

    todo!();
    /*
        CAFFE_ENFORCE_EQ(boxes.cols(), 5);

      // Scale min_size to match image scale
      min_size *= im_info[2];

      using T = typename Derived::Scalar;

      const auto& x_ctr = boxes.col(0);
      const auto& y_ctr = boxes.col(1);
      const auto& ws = boxes.col(2);
      const auto& hs = boxes.col(3);

      EArrXb keep = (ws >= min_size) && (hs >= min_size) &&
          (x_ctr < T(im_info[1])) && (y_ctr < T(im_info[0]));

      return GetArrayIndices(keep);
    */
}
