crate::ix!();

/**
  | Only keep boxes with both sides >= min_size
  | and center within the image.
  | 
  | boxes: pixel coordinates of bounding
  | box, size (M * 4)
  | 
  | im_info: [height, width, img_scale]
  | 
  | return: row indices for 'boxes'
  |
  */
#[inline] pub fn filter_boxes_upright<Derived>(
    boxes:           &ArrayBase<Derived>,
    min_size:        f64,
    im_info:         &Array3f,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(boxes.cols(), 4);

      // Scale min_size to match image scale
      min_size *= im_info[2];

      using T = typename Derived::Scalar;
      using EArrX = EArrXt<T>;

      EArrX ws = boxes.col(2) - boxes.col(0) + T(int(legacy_plus_one));
      EArrX hs = boxes.col(3) - boxes.col(1) + T(int(legacy_plus_one));
      EArrX x_ctr = boxes.col(0) + ws / T(2);
      EArrX y_ctr = boxes.col(1) + hs / T(2);

      EArrXb keep = (ws >= min_size) && (hs >= min_size) &&
          (x_ctr < T(im_info[1])) && (y_ctr < T(im_info[0]));

      return GetArrayIndices(keep);
    */
}
