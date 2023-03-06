crate::ix!();

#[inline] pub fn soft_nms_cpu_with_indices<Derived1, Derived2, Derived3>(
    out_scores:      *mut ArrayBase<Derived3>,
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    indices:         &Vec<i32>,
    sigma:           Option<f32>,
    overlap_thresh:  Option<f32>,
    score_thresh:    Option<f32>,
    method:          Option<u32>,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> 
{
    let sigma:             f32 = sigma.unwrap_or(0.5);
    let overlap_thresh:    f32 = overlap_thresh.unwrap_or(0.3);
    let score_thresh:      f32 = score_thresh.unwrap_or(0.001);
    let method:            u32 = method.unwrap_or(1);
    let topn:              i32 = topn.unwrap_or(-1);
    let legacy_plus_one:   bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
      if (proposals.cols() == 4) {
        // Upright boxes
        return soft_nms_cpu_upright(
            out_scores,
            proposals,
            scores,
            indices,
            sigma,
            overlap_thresh,
            score_thresh,
            method,
            topN,
            legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return soft_nms_cpu_rotated(
            out_scores,
            proposals,
            scores,
            indices,
            sigma,
            overlap_thresh,
            score_thresh,
            method,
            topN);
      }
    */
}
