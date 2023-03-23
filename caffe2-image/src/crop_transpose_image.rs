crate::ix!();

/**
  | Only crop / transpose the image leave
  | in uint8_t dataType
  |
  */
#[inline] pub fn crop_transpose_image<Context>(
    scaled_img:          &cv::Mat,
    channels:            i32,
    cropped_data:        *mut u8,
    crop:                i32,
    mirror:              bool,
    randgen:             *mut mt19937::MT19937,
    mirror_this_image:   *mut statrs::distribution::Bernoulli,
    is_test:             bool) 
{
    todo!();
    /*
        CAFFE_ENFORCE_GE(
          scaled_img.rows, crop, "Image height must be bigger than crop.");
      CAFFE_ENFORCE_GE(
          scaled_img.cols, crop, "Image width must be bigger than crop.");

      // find the cropped region, and copy it to the destination matrix
      int width_offset, height_offset;
      if (is_test) {
        width_offset = (scaled_img.cols - crop) / 2;
        height_offset = (scaled_img.rows - crop) / 2;
      } else {
        width_offset =
            std::uniform_int_distribution<>(0, scaled_img.cols - crop)(*randgen);
        height_offset =
            std::uniform_int_distribution<>(0, scaled_img.rows - crop)(*randgen);
      }

      if (mirror && (*mirror_this_image)(*randgen)) {
        // Copy mirrored image.
        for (int h = height_offset; h < height_offset + crop; ++h) {
          for (int w = width_offset + crop - 1; w >= width_offset; --w) {
            const uint8_t* cv_data = scaled_img.ptr(h) + w * channels;
            for (int c = 0; c < channels; ++c) {
              *(cropped_data++) = cv_data[c];
            }
          }
        }
      } else {
        // Copy normally.
        for (int h = height_offset; h < height_offset + crop; ++h) {
          for (int w = width_offset; w < width_offset + crop; ++w) {
            const uint8_t* cv_data = scaled_img.ptr(h) + w * channels;
            for (int c = 0; c < channels; ++c) {
              *(cropped_data++) = cv_data[c];
            }
          }
        }
      }
    */
}
