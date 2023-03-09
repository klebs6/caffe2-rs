crate::ix!();

#[inline] pub fn bilinear_interpolate_gradient<T>(
    height:   i32,
    width:    i32,
    y:        T,
    x:        T,
    w1:       &mut T,
    w2:       &mut T,
    w3:       &mut T,
    w4:       &mut T,
    x_low:    &mut i32,
    x_high:   &mut i32,
    y_low:    &mut i32,
    y_high:   &mut i32,

    /* index for debug only*/
    index:    i32 ) 
{
    todo!();
    /*
        // deal with cases that inverse elements are out of feature map boundary
      if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
      }

      if (y <= 0) {
        y = 0;
      }
      if (x <= 0) {
        x = 0;
      }

      y_low = (int)y;
      x_low = (int)x;

      if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
      } else {
        y_high = y_low + 1;
      }

      if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
      } else {
        x_high = x_low + 1;
      }

      T ly = y - y_low;
      T lx = x - x_low;
      T hy = 1. - ly, hx = 1. - lx;

      // reference in forward
      // T v1 = bottom_data[y_low * width + x_low];
      // T v2 = bottom_data[y_low * width + x_high];
      // T v3 = bottom_data[y_high * width + x_low];
      // T v4 = bottom_data[y_high * width + x_high];
      // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

      w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

      return;
    */
}
