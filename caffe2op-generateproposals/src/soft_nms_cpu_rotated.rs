crate::ix!();

/**
  | Similar to soft_nms_cpu_upright,
  | but handles rotated proposal boxes
  | in the format:
  | 
  | size (M, 5), format [ctr_x; ctr_y; width;
  | height; angle (in degrees)].
  | 
  | For now, we only consider IoU as the metric
  | for suppression. No angle info is used
  | yet.
  |
  */
#[inline] pub fn soft_nms_cpu_rotated<Derived1, Derived2, Derived3>(
    out_scores:     *mut ArrayBase<Derived3>,
    proposals:      &ArrayBase<Derived1>,
    scores:         &ArrayBase<Derived2>,
    indices:        &Vec<i32>,
    sigma:          Option<f32>,
    overlap_thresh: Option<f32>,
    score_thresh:   Option<f32>,
    method:         Option<u32>,
    topn:           Option<i32>) -> Vec<i32> {

    let sigma: f32 = sigma.unwrap_or(0.5);
    let overlap_thresh: f32 = overlap_thresh.unwrap_or(0.3);
    let score_thresh: f32 = score_thresh.unwrap_or(0.001);
    let method: u32 = method.unwrap_or(1);
    let topn: i32 = topn.unwrap_or(-1);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
      CAFFE_ENFORCE_EQ(proposals.cols(), 5);
      CAFFE_ENFORCE_EQ(scores.cols(), 1);

      using EArrX = EArrXt<typename Derived1::Scalar>;

      auto widths = proposals.col(2);
      auto heights = proposals.col(3);
      EArrX areas = widths * heights;

      std::vector<RotatedRect> rotated_rects(proposals.rows());
      for (int i = 0; i < proposals.rows(); ++i) {
        rotated_rects[i] = bbox_to_rotated_rect(proposals.row(i));
      }

      // Initialize out_scores with original scores. Will be iteratively updated
      // as Soft-NMS is applied.
      *out_scores = scores;

      std::vector<int> keep;
      EArrXi pending = AsEArrXt(indices);
      while (pending.size() > 0) {
        // Exit if already enough proposals
        if (topN >= 0 && keep.size() >= topN) {
          break;
        }

        // Find proposal with max score among remaining proposals
        int max_pos;
        auto max_score = GetSubArray(*out_scores, pending).maxCoeff(&max_pos);
        int i = pending[max_pos];
        keep.push_back(i);

        // Compute IoU of the remaining boxes with the identified max box
        std::swap(pending(0), pending(max_pos));
        const auto& rest_indices = pending.tail(pending.size() - 1);
        EArrX inter(rest_indices.size());
        for (const auto j : c10::irange(rest_indices.size())) {
          inter[j] = rotated_rect_intersection(
              rotated_rects[i], rotated_rects[rest_indices[j]]);
        }
        EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

        // Update scores based on computed IoU, overlap threshold and NMS method
        // TODO (viswanath): Should angle info be included as well while filtering?
        for (const auto j : c10::irange(rest_indices.size())) {
          typename Derived2::Scalar weight;
          switch (method) {
            case 1: // Linear
              weight = (ovr(j) > overlap_thresh) ? (1.0 - ovr(j)) : 1.0;
              break;
            case 2: // Gaussian
              weight = std::exp(-1.0 * ovr(j) * ovr(j) / sigma);
              break;
            default: // Original NMS
              weight = (ovr(j) > overlap_thresh) ? 0.0 : 1.0;
          }
          (*out_scores)(rest_indices[j]) *= weight;
        }

        // Discard boxes with new scores below min threshold and update pending
        // indices
        const auto& rest_scores = GetSubArray(*out_scores, rest_indices);
        const auto& inds = GetArrayIndices(rest_scores >= score_thresh);
        pending = GetSubArray(rest_indices, AsEArrXt(inds));
      }

      return keep;
    */
}
