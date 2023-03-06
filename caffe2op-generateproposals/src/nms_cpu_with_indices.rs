crate::ix!();

#[inline] pub fn nms_cpu_with_indices<Derived1, Derived2>(
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    sorted_indices:  &Vec<i32>,
    thresh:          f32,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let topn:             i32 = topn.unwrap_or(-1);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
      if (proposals.cols() == 4) {
        // Upright boxes
        return nms_cpu_upright(
            proposals, scores, sorted_indices, thresh, topN, legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return nms_cpu_rotated(proposals, scores, sorted_indices, thresh, topN);
      }
    */
}
