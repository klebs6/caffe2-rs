crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vol2col.h]

pub fn vol2col<T>(
        data_vol:      *const T,
        channels:      i64,
        depth:         i64,
        height:        i64,
        width:         i64,
        depth_col:     i64,
        height_col:    i64,
        width_col:     i64,
        kt:            i64,
        kernel_height: i64,
        kernel_width:  i64,
        pt:            i64,
        ph:            i64,
        pw:            i64,
        dt:            i64,
        dh:            i64,
        dw:            i64,
        dilationt:     i64,
        dilationh:     i64,
        dilationw:     i64,
        data_col:      *mut T)  {

    todo!();
        /*
            i64 c, t, h, w;
      i64 channels_col = channels * kT * kernel_height * kernel_width;
      for (c = 0; c < channels_col; ++c) {
        i64 w_offset = c % kernel_width;
        i64 h_offset = (c / kernel_width) % kernel_height;
        i64 t_offset = (c / kernel_width / kernel_height) % kT;
        i64 c_vol = c / kT / kernel_height / kernel_width;
        for (t = 0; t < depth_col; ++t) {
          i64 t_pad = t * dT - pT + t_offset * dilationT;
          for (h = 0; h < height_col; ++h) {
            i64 h_pad = h * dH - pH + h_offset * dilationH;
            for (w = 0; w < width_col; ++w) {
              i64 w_pad = w * dW - pW + w_offset * dilationW;
              if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
                  w_pad >= 0 && w_pad < width)
                data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                    data_vol
                        [((c_vol * depth + t_pad) * height + h_pad) * width +
                         w_pad];
              else
                data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                    0;
            }
          }
        }
      }
        */
}

pub fn col2vol<T>(
        data_col:      *const T,
        channels:      i64,
        depth:         i64,
        height:        i64,
        width:         i64,
        out_depth:     i64,
        out_height:    i64,
        out_width:     i64,
        kt:            i64,
        kernel_height: i64,
        kernel_width:  i64,
        pt:            i64,
        ph:            i64,
        pw:            i64,
        dt:            i64,
        dh:            i64,
        dw:            i64,
        dilationt:     i64,
        dilationh:     i64,
        dilationw:     i64,
        data_vol:      *mut T)  {

    todo!();
        /*
            i64 c, t, h, w;
      memset(data_vol, 0, sizeof(T) * depth * height * width * channels);
      i64 depth_col = out_depth;
      i64 height_col = out_height;
      i64 width_col = out_width;
      i64 channels_col = channels * kT * kernel_height * kernel_width;
      for (c = 0; c < channels_col; ++c) {
        i64 w_offset = c % kernel_width;
        i64 h_offset = (c / kernel_width) % kernel_height;
        i64 t_offset = (c / kernel_width / kernel_height) % kT;
        i64 c_vol = c / kT / kernel_height / kernel_width;
        for (t = 0; t < depth_col; ++t) {
          i64 t_pad = t * dT - pT + t_offset * dilationT;
          for (h = 0; h < height_col; ++h) {
            i64 h_pad = h * dH - pH + h_offset * dilationH;
            for (w = 0; w < width_col; ++w) {
              i64 w_pad = w * dW - pW + w_offset * dilationW;
              if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
                  w_pad >= 0 && w_pad < width)
                data_vol
                    [((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
                    data_col
                        [((c * depth_col + t) * height_col + h) * width_col + w];
            }
          }
        }
      }
        */
}
