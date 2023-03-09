crate::ix!();

pub struct BilinearInterpolationParam<T> {
    p1: i64,
    p2: i64,
    p3: i64,
    p4: i64,
    w1: T,
    w2: T,
    w3: T,
    w4: T,
}

#[inline] pub fn make_bilinear_interpolation_params<T>(
    h:             i64,
    w:             i64,
    pooled_h:      i64,
    pooled_w:      i64,
    bin_size_h:    T,
    bin_size_w:    T,
    bin_grid_h:    i64,
    bin_grid_w:    i64,
    roi_start_h:   T,
    roi_start_w:   T) -> Vec<BilinearInterpolationParam<T>> 
{
    todo!();
    /*
        std::vector<BilinearInterpolationParam<T>> params(
          pooled_h * pooled_w * bin_grid_h * bin_grid_w);
      const T ch = bin_size_h / static_cast<T>(bin_grid_h);
      const T cw = bin_size_w / static_cast<T>(bin_grid_w);
      int64_t cnt = 0;
      for (int64_t ph = 0; ph < pooled_h; ++ph) {
        for (int64_t pw = 0; pw < pooled_w; ++pw) {
          for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
            const T yy = roi_start_h + static_cast<T>(ph) * bin_size_h +
                (static_cast<T>(iy) + T(0.5)) * ch;
            if (yy < T(-1) || yy > static_cast<T>(H)) {
              std::memset(params.data() + cnt, 0, bin_grid_w * sizeof(params[0]));
              cnt += bin_grid_w;
              continue;
            }
            for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
              const T xx = roi_start_w + pw * bin_size_w +
                  (static_cast<T>(ix) + T(0.5f)) * cw;
              BilinearInterpolationParam<T>& param = params[cnt++];
              if (xx < T(-1) || xx > static_cast<T>(W)) {
                std::memset(&param, 0, sizeof(param));
                continue;
              }
              const T y = std::min(std::max(yy, T(0)), static_cast<T>(H - 1));
              const T x = std::min(std::max(xx, T(0)), static_cast<T>(W - 1));
              const int64_t yl = static_cast<int64_t>(std::floor(y));
              const int64_t xl = static_cast<int64_t>(std::floor(x));
              const int64_t yh = std::min(yl + 1, H - 1);
              const int64_t xh = std::min(xl + 1, W - 1);
              const T py = y - static_cast<T>(yl);
              const T px = x - static_cast<T>(xl);
              const T qy = T(1) - py;
              const T qx = T(1) - px;
              param.p1 = yl * W + xl;
              param.p2 = yl * W + xh;
              param.p3 = yh * W + xl;
              param.p4 = yh * W + xh;
              param.w1 = qy * qx;
              param.w2 = qy * px;
              param.w3 = py * qx;
              param.w4 = py * px;
            }
          }
        }
      }
      return params;
    */
}
