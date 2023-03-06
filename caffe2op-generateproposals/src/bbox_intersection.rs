crate::ix!();

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
