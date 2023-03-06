crate::ix!();

/**
  | Eigen doesn't seem to support 2d cross
  | product, so we make one here
  |
  */
#[inline] pub fn cross_2d(a: &Vector2f, b: &Vector2f) -> f32 {
    
    todo!();
    /*
        return A.x() * B.y() - B.x() * A.y();
    */
}

#[inline] pub fn polygon_area(q: *const Vector2f, m: &i32) -> f64 {
    
    todo!();
    /*
        if (m <= 2)
        return 0;
      double area = 0;
      for (int i = 1; i < m - 1; i++)
        area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
      return area / 2.0;
    */
}
