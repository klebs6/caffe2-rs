crate::ix!();

/**
  | Soft-NMS implementation as outlined
  | in https://arxiv.org/abs/1704.04503.
  | 
  | Reference: facebookresearch/Detectron/detectron/utils/cython_nms.pyx
  | 
  | out_scores: Output updated scores
  | after applying Soft-NMS
  | 
  | proposals: pixel coordinates of proposed
  | bounding boxes, size: (M, 4), format:
  | [x1; y1; x2; y2] size: (M, 5), format:
  | [ctr_x; ctr_y; w; h; angle (degrees)]
  | for RRPN
  | 
  | scores: scores for each bounding box,
  | size: (M, 1)
  | 
  | indices: Indices to consider within
  | proposals and scores. Can be used to
  | pre-filter proposals/scores based
  | on some threshold.
  | 
  | sigma: Standard deviation for Gaussian
  | 
  | overlap_thresh: Similar to original
  | NMS
  | 
  | score_thresh: If updated score falls
  | below this thresh, discard proposal
  | 
  | method: 0 - Hard (original) NMS, 1 -
  | Linear, 2 - Gaussian
  | 
  | return: row indices of the selected
  | proposals
  |
  */
#[inline] pub fn soft_nms_cpu_upright<Derived1, Derived2, Derived3>(
    out_scores:      *mut ArrayBase<Derived3>,
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    indices:         &Vec<i32>,
    sigma:           Option<f32>,
    overlap_thresh:  Option<f32>,
    score_thresh:    Option<f32>,
    method:          Option<u32>,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let sigma:              f32 = sigma.unwrap_or(0.5);
    let overlap_thresh:     f32 = overlap_thresh.unwrap_or(0.3);
    let score_thresh:       f32 = score_thresh.unwrap_or(0.001);
    let method:             u32 = method.unwrap_or(1);
    let topn:               i32 = topn.unwrap_or(-1);
    let legacy_plus_one:    bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
      CAFFE_ENFORCE_EQ(proposals.cols(), 4);
      CAFFE_ENFORCE_EQ(scores.cols(), 1);

      using EArrX = EArrXt<typename Derived1::Scalar>;

      const auto& x1 = proposals.col(0);
      const auto& y1 = proposals.col(1);
      const auto& x2 = proposals.col(2);
      const auto& y2 = proposals.col(3);

      EArrX areas =
          (x2 - x1 + int(legacy_plus_one)) * (y2 - y1 + int(legacy_plus_one));

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
        EArrX xx1 = GetSubArray(x1, rest_indices).cwiseMax(x1[i]);
        EArrX yy1 = GetSubArray(y1, rest_indices).cwiseMax(y1[i]);
        EArrX xx2 = GetSubArray(x2, rest_indices).cwiseMin(x2[i]);
        EArrX yy2 = GetSubArray(y2, rest_indices).cwiseMin(y2[i]);

        EArrX w = (xx2 - xx1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX h = (yy2 - yy1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX inter = w * h;
        EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

        // Update scores based on computed IoU, overlap threshold and NMS method
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
