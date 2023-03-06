crate::ix!();

/**
  | Greedy non-maximum suppression for
  | proposed bounding boxes
  | 
  | Reject a bounding box if its region has
  | an intersection-overunion (IoU) overlap
  | with a higher scoring selected bounding
  | box larger than a threshold.
  | 
  | Reference: facebookresearch/Detectron/detectron/lib/utils/cython_nms.pyx
  | 
  | proposals: pixel coordinates of proposed
  | bounding boxes,
  | 
  | size: (M, 4), format: [x1; y1; x2; y2]
  | 
  | size: (M, 5), format: [ctr_x; ctr_y;
  | w; h; angle (degrees)] for RRPN
  | 
  | scores: scores for each bounding box,
  | size: (M, 1)
  | 
  | return: row indices of the selected
  | proposals
  |
  */
#[inline] pub fn nms_cpu<Derived1, Derived2>(
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    thres:           f32,
    legacy_plus_one: Option<bool>) -> Vec<i32> 
{
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        std::vector<int> indices(proposals.rows());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(
          indices.data(),
          indices.data() + indices.size(),
          [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

      return nms_cpu(
          proposals,
          scores,
          indices,
          thres,
          -1 /* topN */,
          legacy_plus_one /* legacy_plus_one */);
    */
}
