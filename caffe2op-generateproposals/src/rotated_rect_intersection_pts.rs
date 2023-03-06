crate::ix!();

/**
  | Returns the intersection area of two
  | rotated rectangles.
  |
  */
#[inline] pub fn rotated_rect_intersection(
    rect1: &RotatedRect,
    rect2: &RotatedRect) -> f64 
{
    todo!();
    /*
        // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
      // from rotated_rect_intersection_pts
      Vector2f intersectPts[24], orderedPts[24];
      int num = 0; // number of intersections

      // Find points of intersection

      // TODO: rotated_rect_intersection_pts is a replacement function for
      // cv::rotatedRectangleIntersection, which has a bug due to float underflow
      // For anyone interested, here're the PRs on OpenCV:
      // https://github.com/opencv/opencv/issues/12221
      // https://github.com/opencv/opencv/pull/12222
      // Note: it doesn't matter if #intersections is greater than 8 here
      auto ret = rotated_rect_intersection_pts(rect1, rect2, intersectPts, num);

      if (num > 24) {
        // should never happen
        string msg = "";
        msg += "num_intersections = " + to_string(num);
        msg += "; rect1.center = (" + to_string(rect1.center.x()) + ", " +
            to_string(rect1.center.y()) + "), ";
        msg += "rect1.size = (" + to_string(rect1.size.x()) + ", " +
            to_string(rect1.size.y()) + "), ";
        msg += "rect1.angle = " + to_string(rect1.angle);
        msg += "; rect2.center = (" + to_string(rect2.center.x()) + ", " +
            to_string(rect2.center.y()) + "), ";
        msg += "rect2.size = (" + to_string(rect2.size.x()) + ", " +
            to_string(rect2.size.y()) + "), ";
        msg += "rect2.angle = " + to_string(rect2.angle);
        CAFFE_ENFORCE(num <= 24, msg);
      }

      if (num <= 2)
        return 0.0;

      // If one rectangle is fully enclosed within another, return the area
      // of the smaller one early.
      if (ret == INTERSECT_FULL) {
        return std::min(
            rect1.size.x() * rect1.size.y(), rect2.size.x() * rect2.size.y());
      }

      // Convex Hull to order the intersection points in clockwise or
      // counter-clockwise order and find the countour area.
      int num_convex = convex_hull_graham(intersectPts, num, orderedPts, true);
      return polygon_area(orderedPts, num_convex);
    */
}


/**
  | rotated_rect_intersection_pts is
  | a replacement function for
  | 
  | cv::rotatedRectangleIntersection,
  | which has a bug due to float underflow
  | 
  | For anyone interested, here're the
  | PRs on OpenCV:
  | 
  | https://github.com/opencv/opencv/issues/12221
  | https://github.com/opencv/opencv/pull/12222
  | 
  | -----------
  | @note
  | 
  | we do not check if the number of intersections
  | is <= 8 in this case
  |
  */
#[inline] pub fn rotated_rect_intersection_pts(
    rect1:         &RotatedRect,
    rect2:         &RotatedRect,
    intersections: *mut Vector2f,
    num:           &mut i32) -> i32 {
    
    todo!();
    /*
        // Used to test if two points are the same
      const float samePointEps = 0.00001f;
      const float EPS = 1e-14;
      num = 0; // number of intersections

      Vector2f vec1[4], vec2[4], pts1[4], pts2[4];

      rect1.get_vertices(pts1);
      rect2.get_vertices(pts2);

      // Specical case of rect1 == rect2
      bool same = true;

      for (int i = 0; i < 4; i++) {
        if (fabs(pts1[i].x() - pts2[i].x()) > samePointEps ||
            (fabs(pts1[i].y() - pts2[i].y()) > samePointEps)) {
          same = false;
          break;
        }
      }

      if (same) {
        for (int i = 0; i < 4; i++) {
          intersections[i] = pts1[i];
        }
        num = 4;
        return INTERSECT_FULL;
      }

      // Line vector
      // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
      for (int i = 0; i < 4; i++) {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
      }

      // Line test - test all line combos for intersection
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          // Solve for 2x2 Ax=b

          // This takes care of parallel lines
          float det = cross_2d(vec2[j], vec1[i]);
          if (std::fabs(det) <= EPS) {
            continue;
          }

          auto vec12 = pts2[j] - pts1[i];

          float t1 = cross_2d(vec2[j], vec12) / det;
          float t2 = cross_2d(vec1[i], vec12) / det;

          if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
            intersections[num++] = pts1[i] + t1 * vec1[i];
          }
        }
      }

      // Check for vertices from rect1 inside rect2
      {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = AB.squaredNorm();
        auto ADdotAD = DA.squaredNorm();
        for (int i = 0; i < 4; i++) {
          // assume ABCD is the rectangle, and P is the point to be judged
          // P is inside ABCD iff. P's projection on AB lies within AB
          // and P's projection on AD lies within AD

          auto AP = pts1[i] - pts2[0];

          auto APdotAB = AP.dot(AB);
          auto APdotAD = -AP.dot(DA);

          if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
              (APdotAD <= ADdotAD)) {
            intersections[num++] = pts1[i];
          }
        }
      }

      // Reverse the check - check for vertices from rect2 inside rect1
      {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = AB.squaredNorm();
        auto ADdotAD = DA.squaredNorm();
        for (int i = 0; i < 4; i++) {
          auto AP = pts2[i] - pts1[0];

          auto APdotAB = AP.dot(AB);
          auto APdotAD = -AP.dot(DA);

          if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
              (APdotAD <= ADdotAD)) {
            intersections[num++] = pts2[i];
          }
        }
      }

      return num ? INTERSECT_PARTIAL : INTERSECT_NONE;
    */
}
