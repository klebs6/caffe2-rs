crate::ix!();

/// Inception-stype scale jittering
#[inline] pub fn random_sized_cropping<Context>(
    img:     *mut cv::Mat,
    crop:    i32,
    randgen: *mut mt19937::MT19937) -> bool 
{
    todo!();
    /*
        cv::Mat scaled_img;
      bool inception_scale_jitter = false;
      int im_height = img->rows, im_width = img->cols;
      int area = im_height * im_width;
      std::uniform_real_distribution<> area_dis(0.08, 1.0);
      std::uniform_real_distribution<> aspect_ratio_dis(3.0 / 4.0, 4.0 / 3.0);

      cv::Mat cropping;
      for (int i = 0; i < 10; ++i) {
        int target_area = int(ceil(area_dis(*randgen) * area));
        float aspect_ratio = aspect_ratio_dis(*randgen);
        int nh = floor(std::sqrt(((float)target_area / aspect_ratio)));
        int nw = floor(std::sqrt(((float)target_area * aspect_ratio)));
        if (nh >= 1 && nh <= im_height && nw >= 1 && nw <= im_width) {
          int height_offset =
              std::uniform_int_distribution<>(0, im_height - nh)(*randgen);
          int width_offset =
              std::uniform_int_distribution<>(0, im_width - nw)(*randgen);
          cv::Rect ROI(width_offset, height_offset, nw, nh);
          cropping = (*img)(ROI);
          cv::resize(
              cropping, scaled_img, cv::Size(crop, crop), 0, 0, cv::INTER_AREA);
          *img = scaled_img;
          inception_scale_jitter = true;
          break;
        }
      }
      return inception_scale_jitter;
    */
}

