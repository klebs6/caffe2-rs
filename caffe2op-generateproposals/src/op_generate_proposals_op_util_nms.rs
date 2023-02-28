crate::ix!();

use crate::{
    ArrayBase,
    Vector2f,
};

/**
  | Greedy non-maximum suppression for
  | proposed bounding boxes
  | 
  | Reject a bounding box if its region has
  | an intersection-overunion (IoU) overlap
  | with a higher scoring selected bounding
  | box larger than a threshold.
  | 
  | Reference: facebookresearch/Detectron/detectron/utils/cython_nms.pyx
  | 
  | proposals: pixel coordinates of proposed
  | bounding boxes, size: (M, 4), format:
  | [x1; y1; x2; y2]
  | 
  | scores: scores for each bounding box,
  | size: (M, 1)
  | 
  | sorted_indices: indices that sorts
  | the scores from high to low
  | 
  | return: row indices of the selected
  | proposals
  |
  */
#[inline] pub fn nms_cpu_upright<Derived1, Derived2>(
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    sorted_indices:  &Vec<i32>,
    thresh:          f32,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let topn:             i32 = topn.unwrap_or(-1);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
      CAFFE_ENFORCE_EQ(proposals.cols(), 4);
      CAFFE_ENFORCE_EQ(scores.cols(), 1);
      CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

      using EArrX = EArrXt<typename Derived1::Scalar>;

      auto x1 = proposals.col(0);
      auto y1 = proposals.col(1);
      auto x2 = proposals.col(2);
      auto y2 = proposals.col(3);

      EArrX areas =
          (x2 - x1 + int(legacy_plus_one)) * (y2 - y1 + int(legacy_plus_one));

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
        EArrX xx1 = GetSubArray(x1, rest_indices).cwiseMax(x1[i]);
        EArrX yy1 = GetSubArray(y1, rest_indices).cwiseMax(y1[i]);
        EArrX xx2 = GetSubArray(x2, rest_indices).cwiseMin(x2[i]);
        EArrX yy2 = GetSubArray(y2, rest_indices).cwiseMin(y2[i]);

        EArrX w = (xx2 - xx1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX h = (yy2 - yy1 + int(legacy_plus_one)).cwiseMax(0.0);
        EArrX inter = w * h;
        EArrX ovr = inter / (areas[i] + GetSubArray(areas, rest_indices) - inter);

        // indices for sub array order[1:n]
        auto inds = GetArrayIndices(ovr <= thresh);
        order = GetSubArray(order, AsEArrXt(inds) + 1);
      }

      return keep;
    */
}

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

pub const INTERSECT_NONE:    i32 = 0;
pub const INTERSECT_PARTIAL: i32 = 1;
pub const INTERSECT_FULL:    i32 = 2;

///------------------------------------------
pub struct RotatedRect {
    center:  Vector2f,
    size:    Vector2f,
    angle:   f32,
}

impl RotatedRect {

    pub fn new(
        p_center: &Vector2f,
        p_size:   &Vector2f,
        p_angle:  f32) -> Self {
    
        todo!();
        /*
            : center(p_center), size(p_size), angle(p_angle)
        */
    }
    
    #[inline] pub fn get_vertices(&self, pt: *mut Vector2f)  {
        
        todo!();
        /*
            // M_PI / 180. == 0.01745329251
        double _angle = angle * 0.01745329251;
        float b = (float)cos(_angle) * 0.5f;
        float a = (float)sin(_angle) * 0.5f;

        pt[0].x() = center.x() - a * size.y() - b * size.x();
        pt[0].y() = center.y() + b * size.y() - a * size.x();
        pt[1].x() = center.x() + a * size.y() - b * size.x();
        pt[1].y() = center.y() - b * size.y() - a * size.x();
        pt[2] = 2 * center - pt[0];
        pt[3] = 2 * center - pt[1];
        */
    }
}

///------------------------------------------
#[inline] pub fn bbox_to_rotated_rect<Derived>(bbox: &ArrayBase<Derived>) -> RotatedRect {

    todo!();
    /*
        CAFFE_ENFORCE_EQ(box.size(), 5);
      // cv::RotatedRect takes angle to mean clockwise rotation, but RRPN bbox
      // representation means counter-clockwise rotation.
      return RotatedRect(
          Vector2f(box[0], box[1]),
          Vector2f(box[2], box[3]),
          -box[4]);
    */
}

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
  | Find the intersection area of two rotated
  | boxes represented in format
  | 
  | [ctr_x, ctr_y, width, height, angle].
  | 
  | `angle` represents counter-clockwise
  | rotation in degrees.
  |
  */
#[inline] pub fn bbox_intersection_rotated<Derived1, Derived2>(
    box1: &ArrayBase<Derived1>, 
    box2: &ArrayBase<Derived2>) -> f64 
{
    todo!();
    /*
        CAFFE_ENFORCE(box1.size() == 5 && box2.size() == 5);
      const auto& rect1 = bbox_to_rotated_rect(box1);
      const auto& rect2 = bbox_to_rotated_rect(box2);
      return rotated_rect_intersection(rect1, rect2);
    */
}

/**
  | Similar to `bbox_overlaps()` in detectron/utils/cython_bbox.pyx,
  | but handles rotated boxes represented
  | in format
  | 
  | [ctr_x, ctr_y, width, height, angle].
  | 
  | `angle` represents counter-clockwise
  | rotation in degrees.
  |
  */
#[inline] pub fn bbox_overlaps_rotated<Derived1, Derived2>(
    boxes: &ArrayBase<Derived1>, 
    query_boxes: &ArrayBase<Derived2>)
{
    todo!();
    /*
        CAFFE_ENFORCE(boxes.cols() == 5 && query_boxes.cols() == 5);

      const auto& boxes_areas = boxes.col(2) * boxes.col(3);
      const auto& query_boxes_areas = query_boxes.col(2) * query_boxes.col(3);

      ArrayXXf overlaps(boxes.rows(), query_boxes.rows());
      for (int i = 0; i < boxes.rows(); ++i) {
        for (int j = 0; j < query_boxes.rows(); ++j) {
          auto inter = bbox_intersection_rotated(boxes.row(i), query_boxes.row(j));
          overlaps(i, j) = (inter == 0.0)
              ? 0.0
              : inter / (boxes_areas[i] + query_boxes_areas[j] - inter);
        }
      }
      return overlaps;
    */
}

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

#[inline] pub fn nms_cpu_with_indices<Derived1, Derived2>(
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    sorted_indices:  &Vec<i32>,
    thresh:          f32,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let topn:             i32 = topn.unwrap_or(-1);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
      if (proposals.cols() == 4) {
        // Upright boxes
        return nms_cpu_upright(
            proposals, scores, sorted_indices, thresh, topN, legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return nms_cpu_rotated(proposals, scores, sorted_indices, thresh, topN);
      }
    */
}

/**
  | Greedy non-maximum suppression for
  | proposed bounding boxes
  | 
  | Reject a bounding box if its region has
  | an intersection-overunion (IoU) overlap
  | with a higher scoring selected bounding
  | box larger than a threshold.
  | 
  | Reference: facebookresearch/Detectron/detectron/lib/utils/cython_nms.pyx
  | 
  | proposals: pixel coordinates of proposed
  | bounding boxes,
  | 
  | size: (M, 4), format: [x1; y1; x2; y2]
  | 
  | size: (M, 5), format: [ctr_x; ctr_y;
  | w; h; angle (degrees)] for RRPN
  | 
  | scores: scores for each bounding box,
  | size: (M, 1)
  | 
  | return: row indices of the selected
  | proposals
  |
  */
#[inline] pub fn nms_cpu<Derived1, Derived2>(
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    thres:           f32,
    legacy_plus_one: Option<bool>) -> Vec<i32> 
{
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        std::vector<int> indices(proposals.rows());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(
          indices.data(),
          indices.data() + indices.size(),
          [&scores](int lhs, int rhs) { return scores(lhs) > scores(rhs); });

      return nms_cpu(
          proposals,
          scores,
          indices,
          thres,
          -1 /* topN */,
          legacy_plus_one /* legacy_plus_one */);
    */
}


#[inline] pub fn soft_nms_cpu_with_indices<Derived1, Derived2, Derived3>(
    out_scores:      *mut ArrayBase<Derived3>,
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    indices:         &Vec<i32>,
    sigma:           Option<f32>,
    overlap_thresh:  Option<f32>,
    score_thresh:    Option<f32>,
    method:          Option<u32>,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> 
{
    let sigma:             f32 = sigma.unwrap_or(0.5);
    let overlap_thresh:    f32 = overlap_thresh.unwrap_or(0.3);
    let score_thresh:      f32 = score_thresh.unwrap_or(0.001);
    let method:            u32 = method.unwrap_or(1);
    let topn:              i32 = topn.unwrap_or(-1);
    let legacy_plus_one:   bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE(proposals.cols() == 4 || proposals.cols() == 5);
      if (proposals.cols() == 4) {
        // Upright boxes
        return soft_nms_cpu_upright(
            out_scores,
            proposals,
            scores,
            indices,
            sigma,
            overlap_thresh,
            score_thresh,
            method,
            topN,
            legacy_plus_one);
      } else {
        // Rotated boxes with angle info
        return soft_nms_cpu_rotated(
            out_scores,
            proposals,
            scores,
            indices,
            sigma,
            overlap_thresh,
            score_thresh,
            method,
            topN);
      }
    */
}

#[inline] pub fn soft_nms_cpu<Derived1, Derived2, Derived3>(
    out_scores:      *mut ArrayBase<Derived3>,
    proposals:       &ArrayBase<Derived1>,
    scores:          &ArrayBase<Derived2>,
    sigma:           Option<f32>,
    overlap_thresh:  Option<f32>,
    score_thresh:    Option<f32>,
    method:          Option<u32>,
    topn:            Option<i32>,
    legacy_plus_one: Option<bool>) -> Vec<i32> {

    let sigma: f32            = sigma.unwrap_or(0.5);
    let overlap_thresh: f32   = overlap_thresh.unwrap_or(0.3);
    let score_thresh: f32     = score_thresh.unwrap_or(0.001);
    let method: u32           = method.unwrap_or(1);
    let topn: i32             = topn.unwrap_or(-1);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        std::vector<int> indices(proposals.rows());
      std::iota(indices.begin(), indices.end(), 0);
      return soft_nms_cpu(
          out_scores,
          proposals,
          scores,
          indices,
          sigma,
          overlap_thresh,
          score_thresh,
          method,
          topN,
          legacy_plus_one);
    */
}
