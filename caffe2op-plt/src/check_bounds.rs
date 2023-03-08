crate::ix!();

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {

    #[inline] pub fn check_bounds_sorted(
        &mut self, 
        bounds:                *const T,
        num_bounds_per_group:  i64,
        num_group:             i64) -> bool 
    {
        
        todo!();
        /*
            const T* start = bounds;
        for (int64_t i = 0; i < num_group; i++) {
          if (!std::is_sorted(start, start + num_bounds_per_group)) {
            return false;
          }
          start += num_bounds_per_group;
        }
        return true;
        */
    }

    /**
      | Returns true if the transform params from
      | arg are valid.
      |
      | Otherwise, we will assume the transform
      | params will pass from Input blobs.
      */
    #[inline] pub fn check_trans_param_from_arg(&mut self) -> bool {
        
        todo!();
        /*
            int good_param = 0;
        good_param += bounds_from_arg_.size() > 0;
        good_param += slopes_from_arg_.size() > 0;
        good_param += intercepts_from_arg_.size() > 0;
        CAFFE_ENFORCE(
            good_param == 0 || good_param == 3,
            "bounds, slopes, intercepts must be all set or all not set");
        if (good_param == 3) {
          int64_t num_func_per_group;
          int64_t num_group;
          InferNumFunctionsPerGroup(
              bounds_from_arg_.size(),
              slopes_from_arg_.size(),
              intercepts_from_arg_.size(),
              &num_func_per_group,
              &num_group);
          CAFFE_ENFORCE(
              CheckBoundsSorted(
                  bounds_from_arg_.data(), num_func_per_group + 1, num_group),
              "bounds must be sorted for each group");
        }

        return good_param == 3;
        */
    }
}
