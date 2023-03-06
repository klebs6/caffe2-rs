crate::ix!();

/**
  | Similar to nms_cpu_upright, but handles
  | rotated proposal boxes in the format:
  | 
  | size (M, 5), format [ctr_x; ctr_y; width;
  | height; angle (in degrees)].
  | 
  | For now, we only consider IoU as the metric
  | for suppression. No angle info is used
  | yet.
  |
  */
#[inline] pub fn nms_cpu_rotated<Derived1, Derived2>(
    proposals:      &ArrayBase<Derived1>,
    scores:         &ArrayBase<Derived2>,
    sorted_indices: &Vec<i32>,
    thresh:         f32,
    topn:           Option<i32>) -> Vec<i32> {

    let topn: i32 = topn.unwrap_or(-1);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
      CAFFE_ENFORCE_EQ(proposals.cols(), 5);
      CAFFE_ENFORCE_EQ(scores.cols(), 1);
      CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

      using EArrX = EArrXt<typename Derived1::Scalar>;

      auto widths = proposals.col(2);
      auto heights = proposals.col(3);
      EArrX areas = widths * heights;

      std::vector<RotatedRect> rotated_rects(proposals.rows());
      for (int i = 0; i < proposals.rows(); ++i) {
        rotated_rects[i] = bbox_to_rotated_rect(proposals.row(i));
      }

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

        EArrX inter(rest_indices.size());
        for (const auto j : c10::irange(rest_indices.size())) {
          inter[j] = rotated_rect_intersection(
              rotated_rects[i], rotated_rects[rest_indices[j]]);
        }
        EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

        // indices for sub array order[1:n].
        // TODO (viswanath): Should angle info be included as well while filtering?
        auto inds = GetArrayIndices(ovr <= thresh);
        order = GetSubArray(order, AsEArrXt(inds) + 1);
      }

      return keep;
    */
}

