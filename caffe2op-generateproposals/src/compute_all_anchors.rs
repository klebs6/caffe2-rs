crate::ix!();

/**
  | Generate a list of bounding box shapes
  | for each pixel based on predefined bounding
  | box shapes 'anchors'.
  | 
  | anchors: predefined anchors, size(A, 4)
  | 
  | Return: all_anchors_vec: (H * W, A * 4)
  | 
  | Need to reshape to (H * W * A, 4) to match
  | the format in python
  |
  */
#[inline] pub fn compute_all_anchors(
    anchors:     &TensorCPU,
    height:      i32,
    width:       i32,
    feat_stride: f32)  {
    
    todo!();
    /*
        const auto K = height * width;
      const auto A = anchors.size(0);
      const auto box_dim = anchors.size(1);
      CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

      ERMatXf shift_x = (ERVecXf::LinSpaced(width, 0.0, width - 1.0) * feat_stride)
                            .replicate(height, 1);
      ERMatXf shift_y = (EVecXf::LinSpaced(height, 0.0, height - 1.0) * feat_stride)
                            .replicate(1, width);
      Eigen::MatrixXf shifts(K, box_dim);
      if (box_dim == 4) {
        // Upright boxes in [x1, y1, x2, y2] format
        shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
            ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
            ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
            ConstEigenVectorMap<float>(shift_y.data(), shift_y.size());
      } else {
        // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
        // Zero shift for width, height and angle.
        ERMatXf shift_zero = ERMatXf::Constant(height, width, 0.0);
        shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
            ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
            ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
            ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
            ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size());
      }

      // Broacast anchors over shifts to enumerate all anchors at all positions
      // in the (H, W) grid:
      //   - add A anchors of shape (1, A, box_dim) to
      //   - K shifts of shape (K, 1, box_dim) to get
      //   - all shifted anchors of shape (K, A, box_dim)
      //   - reshape to (K*A, box_dim) shifted anchors
      ConstEigenMatrixMap<float> anchors_vec(
          anchors.template data<float>(), 1, A * box_dim);
      // equivalent to python code
      //  all_anchors = (
      //        self._model.anchors.reshape((1, A, box_dim)) +
      //        shifts.reshape((1, K, box_dim)).transpose((1, 0, 2)))
      //    all_anchors = all_anchors.reshape((K * A, box_dim))
      // all_anchors_vec: (K, A * box_dim)
      ERMatXf all_anchors_vec =
          anchors_vec.replicate(K, 1) + shifts.rowwise().replicate(A);

      // use the following to reshape to (K * A, box_dim)
      // Eigen::Map<const ERMatXf> all_anchors(
      //            all_anchors_vec.data(), K * A, box_dim);

      return all_anchors_vec;
    */
}
