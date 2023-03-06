crate::ix!();

/**
 | Generate bounding box proposals for a given image
 |
 | im_info: [height, width, im_scale]
 | all_anchors: (H * W * A, 4)
 | bbox_deltas_tensor: (4 * A, H, W)
 | scores_tensor: (A, H, W)
 | out_boxes: (n, 5)
 | out_probs: n
 */
#[inline] pub fn proposals_for_one_image(
    im_info:            &Array3f,
    anchors:            &ERArrXXf,
    bbox_deltas_tensor: &ConstTensorView<f32>,
    scores_tensor:      &ConstTensorView<f32>,
    out_boxes:          *mut ERArrXXf,
    out_probs:          *mut EArrXf)  {
    
    todo!();
    /*
        const auto& post_nms_topN = rpn_post_nms_topN_;
      const auto& nms_thresh = rpn_nms_thresh_;
      const auto& min_size = rpn_min_size_;
      const int box_dim = static_cast<int>(anchors.cols());
      CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

      CAFFE_ENFORCE_EQ(bbox_deltas_tensor.ndim(), 3);
      CAFFE_ENFORCE_EQ(bbox_deltas_tensor.dim(0) % box_dim, 0);
      auto A = bbox_deltas_tensor.dim(0) / box_dim;
      auto H = bbox_deltas_tensor.dim(1);
      auto W = bbox_deltas_tensor.dim(2);
      auto K = H * W;
      CAFFE_ENFORCE_EQ(A, anchors.rows());

      // scores are (A, H, W) format from conv output.
      // Maintain the same order without transposing (which is slow)
      // and compute anchors accordingly.
      CAFFE_ENFORCE_EQ(scores_tensor.ndim(), 3);
      CAFFE_ENFORCE_EQ(scores_tensor.dims(), (vector<int>{A, H, W}));
      Eigen::Map<const EArrXf> scores(scores_tensor.data(), scores_tensor.size());

      std::vector<int> order(scores.size());
      std::iota(order.begin(), order.end(), 0);
      if (rpn_pre_nms_topN_ <= 0 || rpn_pre_nms_topN_ >= scores.size()) {
        // 4. sort all (proposal, score) pairs by score from highest to lowest
        // 5. take top pre_nms_topN (e.g. 6000)
        std::sort(order.begin(), order.end(), [&scores](int lhs, int rhs) {
          return scores[lhs] > scores[rhs];
        });
      } else {
        // Avoid sorting possibly large arrays; First partition to get top K
        // unsorted and then sort just those (~20x faster for 200k scores)
        std::partial_sort(
            order.begin(),
            order.begin() + rpn_pre_nms_topN_,
            order.end(),
            [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
        order.resize(rpn_pre_nms_topN_);
      }

      EArrXf scores_sorted;
      utils::GetSubArray(scores, utils::AsEArrXt(order), &scores_sorted);

      // bbox_deltas are (A * box_dim, H, W) format from conv output.
      // Order them based on scores maintaining the same format without
      // expensive transpose.
      // Note that order corresponds to (A, H * W) in row-major whereas
      // bbox_deltas are in (A, box_dim, H * W) in row-major. Hence, we
      // obtain a sub-view of bbox_deltas for each dim (4 for RPN, 5 for RRPN)
      // in (A, H * W) with an outer stride of box_dim * H * W. Then we apply
      // the ordering and filtering for each dim iteratively.
      ERArrXXf bbox_deltas_sorted(order.size(), box_dim);
      EArrXf bbox_deltas_per_dim(A * K);
      EigenOuterStride stride(box_dim * K);
      for (int j = 0; j < box_dim; ++j) {
        Eigen::Map<ERMatXf>(bbox_deltas_per_dim.data(), A, K) =
            Eigen::Map<const ERMatXf, 0, EigenOuterStride>(
                bbox_deltas_tensor.data() + j * K, A, K, stride);
        for (int i = 0; i < order.size(); ++i) {
          bbox_deltas_sorted(i, j) = bbox_deltas_per_dim[order[i]];
        }
      }

      // Compute anchors specific to the ordered and pre-filtered indices
      // in (A, H, W) format.
      const auto& all_anchors_sorted =
          utils::ComputeSortedAnchors(anchors, H, W, feat_stride_, order);

      // Transform anchors into proposals via bbox transformations
      static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
      auto proposals = utils::bbox_transform(
          all_anchors_sorted,
          bbox_deltas_sorted,
          bbox_weights,
          utils::BBOX_XFORM_CLIP_DEFAULT,
          legacy_plus_one_,
          angle_bound_on_,
          angle_bound_lo_,
          angle_bound_hi_);

      // 2. clip proposals to image (may result in proposals with zero area
      // that will be removed in the next step)
      proposals = utils::clip_boxes(
          proposals, im_info[0], im_info[1], clip_angle_thresh_, legacy_plus_one_);

      // 3. remove predicted boxes with either height or width < min_size
      auto keep =
          utils::filter_boxes(proposals, min_size, im_info, legacy_plus_one_);
      DCHECK_LE(keep.size(), scores_sorted.size());

      // 6. apply loose nms (e.g. threshold = 0.7)
      // 7. take after_nms_topN (e.g. 300)
      // 8. return the top proposals (-> RoIs top)
      if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
        keep = utils::nms_cpu(
            proposals,
            scores_sorted,
            keep,
            nms_thresh,
            post_nms_topN,
            legacy_plus_one_);
      } else {
        keep = utils::nms_cpu(
            proposals, scores_sorted, keep, nms_thresh, -1, legacy_plus_one_);
      }

      // Generate outputs
      utils::GetSubArrayRows(proposals, utils::AsEArrXt(keep), out_boxes);
      utils::GetSubArray(scores_sorted, utils::AsEArrXt(keep), out_probs);
    */
}
