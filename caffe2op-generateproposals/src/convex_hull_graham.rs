crate::ix!();

/**
  | Compute convex hull using Graham scan
  | algorithm
  |
  */
#[inline] pub fn convex_hull_graham(
    p:             *const Vector2f,
    num_in:        &i32,
    q:             *mut Vector2f,
    shift_to_zero: Option<bool>) -> i32 {

    let shift_to_zero: bool = shift_to_zero.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(num_in >= 2);
      std::vector<int> order;

      // Step 1:
      // Find point with minimum y
      // if more than 1 points have the same minimum y,
      // pick the one with the mimimum x.
      int t = 0;
      for (int i = 1; i < num_in; i++) {
        if (p[i].y() < p[t].y() || (p[i].y() == p[t].y() && p[i].x() < p[t].x())) {
          t = i;
        }
      }
      auto& s = p[t]; // starting point

      // Step 2:
      // Subtract starting point from every points (for sorting in the next step)
      for (int i = 0; i < num_in; i++) {
        q[i] = p[i] - s;
      }

      // Swap the starting point to position 0
      std::swap(q[0], q[t]);

      // Step 3:
      // Sort point 1 ~ num_in according to their relative cross-product values
      // (essentially sorting according to angles)
      std::sort(
          q + 1,
          q + num_in,
          [](const Vector2f& A, const Vector2f& B) -> bool {
            float temp = cross_2d(A, B);
            if (fabs(temp) < 1e-6) {
              return A.squaredNorm() < B.squaredNorm();
            } else {
              return temp > 0;
            }
          });

      // Step 4:
      // Make sure there are at least 2 points (that don't overlap with each other)
      // in the stack
      int k; // index of the non-overlapped second point
      for (k = 1; k < num_in; k++) {
        if (q[k].squaredNorm() > 1e-8)
          break;
      }
      if (k == num_in) {
        // We reach the end, which means the convex hull is just one point
        q[0] = p[t];
        return 1;
      }
      q[1] = q[k];
      int m = 2; // 2 elements in the stack
      // Step 5:
      // Finally we can start the scanning process.
      // If we find a non-convex relationship between the 3 points,
      // we pop the previous point from the stack until the stack only has two
      // points, or the 3-point relationship is convex again
      for (int i = k + 1; i < num_in; i++) {
        while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
          m--;
        }
        q[m++] = q[i];
      }

      // Step 6 (Optional):
      // In general sense we need the original coordinates, so we
      // need to shift the points back (reverting Step 2)
      // But if we're only interested in getting the area/perimeter of the shape
      // We can simply return.
      if (!shift_to_zero) {
        for (int i = 0; i < m; i++)
          q[i] += s;
      }

      return m;
    */
}
