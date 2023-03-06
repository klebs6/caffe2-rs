crate::ix!();

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
