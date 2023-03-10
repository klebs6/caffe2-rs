crate::ix!();

/// Compute the area of an array of boxes.
///
#[inline] pub fn boxes_area(boxes: &ERArrXXf, legacy_plus_one: Option<bool>) -> ERArrXXf {

    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);
    
    todo!();
    /*
        // equivalent to python code
      //   w = (boxes[:, 2] - boxes[:, 0] + 1)
      //   h = (boxes[:, 3] - boxes[:, 1] + 1)
      //   areas = w * h
      //   assert np.all(areas >= 0), 'Negative areas founds'
      const auto w = boxes.col(2) - boxes.col(0) + int(legacy_plus_one);
      const auto h = boxes.col(3) - boxes.col(1) + int(legacy_plus_one);
      const ERArrXXf areas = w * h;
      CAFFE_ENFORCE((areas >= 0).all(), "Negative areas founds: ", boxes);
      return areas;
    */
}
