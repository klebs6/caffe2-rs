crate::ix!();

#[inline] pub fn canonical_dims(
    adims: IDEEPTensorDims,
    axis: i32) -> IDEEPTensorDims 
{
    todo!();
    /*
        CAFFE_ENFORCE(axis < (int32_t)adims.size(), "Invalid axis!");
      CAFFE_ENFORCE(axis > (int32_t)-adims.size(), "Invalid axis!");
      if (adims.size() == 2 || axis == 1)
        return adims;
      if (axis < 0) {
        axis += (int32_t)adims.size();
      }

      auto dim0 = std::accumulate(adims.begin(), adims.begin() + axis, 1,
                                  std::multiplies<ideep::tensor::dim_t>());
      auto dim1 = std::accumulate(adims.begin() + axis, adims.end(), 1,
                                  std::multiplies<ideep::tensor::dim_t>());
      return ideep::tensor::dims({dim0, dim1});
    */
}
