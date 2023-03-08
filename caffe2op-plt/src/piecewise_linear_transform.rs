crate::ix!();

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {

    #[inline] pub fn piecewise_linear_transform(
        &mut self, 
        x:                    T,
        bounds:               *const T,
        slopes:               *const T,
        intercepts:           *const T,
        num_func_per_group:   i64) -> T 
    {
        todo!();
        /*
            T y = 0;
        // deal with samples out of bounds
        // make it the same as the upper/lower bound value
        if (x <= bounds[0]) {
          y = slopes[0] * bounds[0] + intercepts[0];
        } else if (x >= bounds[num_func_per_group]) {
          y = slopes[num_func_per_group - 1] * bounds[num_func_per_group] +
              intercepts[num_func_per_group - 1];
        } else {
          auto low_bound =
              std::lower_bound(bounds, bounds + num_func_per_group + 1, x);
          int bounds_idx = low_bound - bounds - 1;
          // compute the piecewise linear transformation as Y
          y = slopes[bounds_idx] * x + intercepts[bounds_idx];
        }
        return y;
        */
    }
}
