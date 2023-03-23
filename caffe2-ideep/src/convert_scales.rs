crate::ix!();

/**
  | Convert zero_point scales to min_max scales
  | NOTE:
  |
  |  The scales in operator is saved in FBGEMM
  |  format, while FBGEMM scales are the
  |  reciprocals of MKL-DNN scales.
  |
  |  This function is provided to convert scales
  |  from FBGEMM to MKL-DNN
  */
#[inline] pub fn convert_scales(scales_z: Vec<f32>) -> IDEEPScale {
    
    todo!();
    /*
        ideep::scale_t scales (scales_z);
      for (auto it = scales.begin(); it != scales.end(); it++) {
        *it = 1.0f / *it;
      }
      return scales;
    */
}
