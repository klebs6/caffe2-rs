crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/im2col.h]

pub fn im2col<T>(
    data_im:       *const T,
    channels:      i64,
    height:        i64,
    width:         i64,
    output_height: i64,
    output_width:  i64,
    kernel_h:      i64,
    kernel_w:      i64,
    pad_h:         i64,
    pad_w:         i64,
    stride_h:      i64,
    stride_w:      i64,
    dilation_h:    i64,
    dilation_w:    i64,
    data_col:      *mut T)  {

    todo!();
        /*
            const i64 height_col = output_height;
      const i64 width_col = output_width;
      const i64 channels_col = channels * kernel_h * kernel_w;

      for (i64 c_col = 0; c_col < channels_col; ++c_col) {
        i64 w_offset = c_col % kernel_w;
        i64 h_offset = (c_col / kernel_w) % kernel_h;
        i64 c_im = c_col / kernel_h / kernel_w;

        for (i64 h_col = 0; h_col < height_col; ++h_col) {
          i64 h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

          for (i64 w_col = 0; w_col < width_col; ++w_col) {
            i64 w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
            data_col[(c_col * height_col + h_col) * width_col + w_col] =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im[(c_im * height + h_im) * width + w_im]
                : static_cast<T>(0);
          }
        }
      }
        */
}

pub fn col2im<T>(
    data_col:      *const T,
    channels:      i64,
    height:        i64,
    width:         i64,
    output_height: i64,
    output_width:  i64,
    kernel_h:      i64,
    kernel_w:      i64,
    pad_h:         i64,
    pad_w:         i64,
    stride_h:      i64,
    stride_w:      i64,
    dilation_h:    i64,
    dilation_w:    i64,
    data_im:       *mut T)  {

    todo!();
        /*
            fill_n(data_im, height * width * channels, T(0));

      const i64 height_col = output_height;
      const i64 width_col = output_width;
      const i64 channels_col = channels * kernel_h * kernel_w;

      for (i64 c_col = 0; c_col < channels_col; ++c_col) {
        i64 w_offset = c_col % kernel_w;
        i64 h_offset = (c_col / kernel_w) % kernel_h;
        i64 c_im = c_col / kernel_h / kernel_w;

        for (i64 h_col = 0; h_col < height_col; ++h_col) {
          i64 h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

          for (i64 w_col = 0; w_col < width_col; ++w_col) {
            i64 w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

            if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
              data_im[(c_im * height + h_im) * width + w_im] +=
                  data_col[(c_col * height_col + h_col) * width_col + w_col];
          }
        }
      }
        */
}
