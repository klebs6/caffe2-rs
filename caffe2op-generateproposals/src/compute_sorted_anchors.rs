crate::ix!();

/**
  | Like ComputeAllAnchors, but instead
  | of computing anchors for every single
  | spatial location, only computes anchors
  | for the already sorted and filtered
  | positions after NMS is applied to avoid
  | unnecessary computation.
  | 
  | `order` is a raveled array of sorted
  | indices in (A, H, W) format.
  |
  */
#[inline] pub fn compute_sorted_anchors(
    anchors:     &ERArrXXf,
    height:      i32,
    width:       i32,
    feat_stride: f32,
    order:       &Vec<i32>) -> ERArrXXf {
    
    todo!();
    /*
        const auto box_dim = anchors.cols();
      CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

      // Order is flattened in (A, H, W) format. Unravel the indices.
      const auto& order_AHW = utils::AsEArrXt(order);
      const auto& order_AH = order_AHW / width;
      const auto& order_W = order_AHW - order_AH * width;
      const auto& order_A = order_AH / height;
      const auto& order_H = order_AH - order_A * height;

      // Generate shifts for each location in the H * W grid corresponding
      // to the sorted scores in (A, H, W) order.
      const auto& shift_x = order_W.cast<float>() * feat_stride;
      const auto& shift_y = order_H.cast<float>() * feat_stride;
      Eigen::MatrixXf shifts(order.size(), box_dim);
      if (box_dim == 4) {
        // Upright boxes in [x1, y1, x2, y2] format
        shifts << shift_x, shift_y, shift_x, shift_y;
      } else {
        // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
        // Zero shift for width, height and angle.
        const auto& shift_zero = EArrXf::Constant(order.size(), 0.0);
        shifts << shift_x, shift_y, shift_zero, shift_zero, shift_zero;
      }

      // Apply shifts to the relevant anchors.
      // Equivalent to python code `all_anchors = self._anchors[order_A] + shifts`
      ERArrXXf anchors_sorted;
      utils::GetSubArrayRows(anchors, order_A, &anchors_sorted);
      const auto& all_anchors_sorted = anchors_sorted + shifts.array();
      return all_anchors_sorted;
    */
}
