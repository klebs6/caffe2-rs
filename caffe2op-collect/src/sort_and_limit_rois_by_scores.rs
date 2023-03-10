crate::ix!();

/**
  | Sort RoIs from highest to lowest individual
  | RoI score based on values from scores
  | array and limit to n results
  |
  */
#[inline] pub fn sort_and_limit_ro_is_by_scores(
    scores: &EArrXf,
    n:      i32,
    rois:   &mut ERArrXXf)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(rois.rows() == scores.size(), "RoIs and scores count mismatch");
      // Create index array with 0, 1, ... N
      std::vector<int> idxs(rois.rows());
      std::iota(idxs.begin(), idxs.end(), 0);
      // Reuse a comparator based on scores and store a copy of RoIs that
      // will be truncated and manipulated below
      auto comp = [&scores](int lhs, int rhs) {
        if (scores(lhs) > scores(rhs)) {
          return true;
        }
        if (scores(lhs) < scores(rhs)) {
          return false;
        }
        // To ensure the sort is stable
        return lhs < rhs;
      };
      ERArrXXf rois_copy = rois;
      // Note that people have found nth_element + sort to be much faster
      // than partial_sort so we use it here
      if (n > 0 && n < rois.rows()) {
        std::nth_element(idxs.begin(), idxs.begin() + n, idxs.end(), comp);
        rois.resize(n, rois.cols());
      } else {
        n = rois.rows();
      }
      std::sort(idxs.begin(), idxs.begin() + n, comp);
      // Update RoIs based on new order
      for (int i = 0; i < n; i++) {
        rois.row(i) = rois_copy.row(idxs[i]);
      }
    */
}
