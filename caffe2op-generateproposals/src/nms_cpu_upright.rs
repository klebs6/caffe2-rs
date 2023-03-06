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
  | Reference: facebookresearch/Detectron/detectron/utils/cython_nms.pyx
  | 
  | proposals: pixel coordinates of proposed
  | bounding boxes, size: (M, 4), format:
  | [x1; y1; x2; y2]
  | 
  | scores: scores for each bounding box,
  | size: (M, 1)
  | 
  | sorted_indices: indices that sorts
  | the scores from high to low
  | 
  | return: row indices of the selected
  | proposals
  |
  */
#[inline] pub fn nms_cpu_upright<Derived1, Derived2>(
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
        CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
      CAFFE_ENFORCE_EQ(proposals.cols(), 4);
      CAFFE_ENFORCE_EQ(scores.cols(), 1);
      CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

      using EArrX = EArrXt<typename Derived1::Scalar>;

      auto x1 = proposals.col(0);
      auto y1 = proposals.col(1);
      auto x2 = proposals.col(2);
      auto y2 = proposals.col(3);

      EArrX areas =
          (x2 - x1 + int(legacy_plus_one)) * (y2 - y1 + int(legacy_plus_one));

      EArrXi order = AsEArrXt(sorted_indices);
      std::vector<int> keep;
      while (order.size() > 0) {
        // exit if already enough proposals
        if (topN >= 0 && keep.size() >= topN) {
          break;
        }

        int i = order[0];
        keep.push_back(i);
        ConstEigenVectorArrayMap<int> rest_indices(
            order.data() + 1, order.size() - 1);
        EArrX xx1 = GetSubArray(x1, rest_indices).cwiseMax(x1[i]);
        EArrX yy1 = GetSubArray(y1, rest_indices).cwiseMax(y1[i]);
        EArrX xx2 = GetSubArray(x2, rest_indices).cwiseMin(x2[i]);
        EArrX yy2 = GetSubArray(y2, rest_indices).cwiseMin(y2[i]);

        EArrX w = (xx2 - xx1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX h = (yy2 - yy1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX inter = w * h;
        EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

        // indices for sub array order[1:n]
        auto inds = GetArrayIndices(ovr <= thresh);
        order = GetSubArray(order, AsEArrXt(inds) + 1);
      }

      return keep;
    */
}
