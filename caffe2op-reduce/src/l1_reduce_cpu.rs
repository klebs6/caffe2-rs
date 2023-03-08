crate::ix!();

impl L1Reducer<CPUContext> {

    #[inline] pub fn backward_cpu<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const float kEps = 1e-12f;
      const auto dX_size = c10::multiply_integers(dX_dims.cbegin(), dX_dims.cend());
      const int ndim = dX_dims.size();
      std::vector<int> index(ndim, 0);
      for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
        const int dY_index =
            math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
        float temp = X_data[dX_index];
        if (temp < -kEps) {
          dX_data[dX_index] = -dY_data[dY_index];
        } else if (temp > kEps) {
          dX_data[dX_index] = dY_data[dY_index];
        } else {
          dX_data[dX_index] = T(0);
        }
        math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
      }
      return true;
        */
    }
}
