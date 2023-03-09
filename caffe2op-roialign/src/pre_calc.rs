crate::ix!();

pub struct PreCalc<T> {
    pos1:  i32,
    pos2:  i32,
    pos3:  i32,
    pos4:  i32,
    w1:    T,
    w2:    T,
    w3:    T,
    w4:    T,
}

#[inline] pub fn pre_calc_for_bilinear_interpolate<T>(
    height:         i32,
    width:          i32,
    pooled_height:  i32,
    pooled_width:   i32,
    iy_upper:       i32,
    ix_upper:       i32,
    roi_start_h:    T,
    roi_start_w:    T,
    bin_size_h:     T,
    bin_size_w:     T,
    roi_bin_grid_h: i32,
    roi_bin_grid_w: i32,
    roi_center_h:   T,
    roi_center_w:   T,
    theta:          T,
    pre_calc:       &mut Vec<PreCalc<T>>)  {

    todo!();
    /*
        int pre_calc_index = 0;
      T cosTheta = cos(theta);
      T sinTheta = sin(theta);
      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          for (int iy = 0; iy < iy_upper; iy++) {
            const T yy = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < ix_upper; ix++) {
              const T xx = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

              // Rotate by theta around the center and translate
              T x = xx * cosTheta + yy * sinTheta + roi_center_w;
              T y = yy * cosTheta - xx * sinTheta + roi_center_h;

              // deal with: inverse elements are out of feature map boundary
              if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                PreCalc<T> pc;
                pc.pos1 = 0;
                pc.pos2 = 0;
                pc.pos3 = 0;
                pc.pos4 = 0;
                pc.w1 = 0;
                pc.w2 = 0;
                pc.w3 = 0;
                pc.w4 = 0;
                pre_calc[pre_calc_index] = pc;
                pre_calc_index += 1;
                continue;
              }

              if (y <= 0) {
                y = 0;
              }
              if (x <= 0) {
                x = 0;
              }

              int y_low = (int)y;
              int x_low = (int)x;
              int y_high;
              int x_high;

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
              T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

              // Save weights and indices
              PreCalc<T> pc;
              pc.pos1 = y_low * width + x_low;
              pc.pos2 = y_low * width + x_high;
              pc.pos3 = y_high * width + x_low;
              pc.pos4 = y_high * width + x_high;
              pc.w1 = w1;
              pc.w2 = w2;
              pc.w3 = w3;
              pc.w4 = w4;
              pre_calc[pre_calc_index] = pc;

              pre_calc_index += 1;
            }
          }
        }
      }
    */
}
