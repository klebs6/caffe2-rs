crate::ix!();

/// Factored out image transformation
#[inline] pub fn transform_image<Context>(
    scaled_img:              &cv::Mat,
    channels:                i32,
    image_data:              *mut f32,
    color_jitter:            bool,
    saturation:              f32,
    brightness:              f32,
    contrast:                f32,
    color_lighting:          bool,
    color_lighting_std:      f32,
    color_lighting_eigvecs:  &Vec<Vec<f32>>,
    color_lighting_eigvals:  &Vec<f32>,
    crop:                    i32,
    mirror:                  bool,
    mean:                    &Vec<f32>,
    std:                     &Vec<f32>,
    randgen:                 *mut mt19937::MT19937,
    mirror_this_image:       *mut statrs::distribution::Bernoulli,
    is_test:                 bool) 
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

      float* image_data_ptr = image_data;
      if (!is_test && mirror && (*mirror_this_image)(*randgen)) {
        // Copy mirrored image.
        for (int h = height_offset; h < height_offset + crop; ++h) {
          for (int w = width_offset + crop - 1; w >= width_offset; --w) {
            const uint8_t* cv_data = scaled_img.ptr(h) + w * channels;
            for (int c = 0; c < channels; ++c) {
              *(image_data_ptr++) = static_cast<float>(cv_data[c]);
            }
          }
        }
      } else {
        // Copy normally.
        for (int h = height_offset; h < height_offset + crop; ++h) {
          for (int w = width_offset; w < width_offset + crop; ++w) {
            const uint8_t* cv_data = scaled_img.ptr(h) + w * channels;
            for (int c = 0; c < channels; ++c) {
              *(image_data_ptr++) = static_cast<float>(cv_data[c]);
            }
          }
        }
      }

      if (color_jitter && channels == 3 && !is_test) {
        ColorJitter<Context>(
            image_data, crop, saturation, brightness, contrast, randgen);
      }
      if (color_lighting && channels == 3 && !is_test) {
        ColorLighting<Context>(
            image_data,
            crop,
            color_lighting_std,
            color_lighting_eigvecs,
            color_lighting_eigvals,
            randgen);
      }

      // Color normalization
      // Mean subtraction and scaling.
      ColorNormalization<Context>(image_data, crop, channels, mean, std);
    */
}

