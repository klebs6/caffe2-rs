crate::ix!();

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {

    #[inline] pub fn get_trans_param_data(&mut self, 
        bounds:              *const *const T,
        slopes:              *const *const T,
        intercepts:          *const *const T,
        num_func_per_group:  *mut i64,
        num_group:           *mut i64)  
    {

        todo!();
        /*
            int64_t num_bounds;
        int64_t num_slopes;
        int64_t num_intercepts;

        if (transform_param_from_arg_) {
          CAFFE_ENFORCE_EQ(InputSize(), 1);
          *bounds = bounds_from_arg_.data();
          *slopes = slopes_from_arg_.data();
          *intercepts = intercepts_from_arg_.data();
          num_bounds = bounds_from_arg_.size();
          num_slopes = slopes_from_arg_.size();
          num_intercepts = intercepts_from_arg_.size();
        } else {
          CAFFE_ENFORCE_EQ(InputSize(), 4);
          auto& bounds_input = Input(BOUNDS);
          auto& slopes_input = Input(SLOPES);
          auto& intercepts_input = Input(INTERCEPTS);
          *bounds = bounds_input.template data<T>();
          *slopes = slopes_input.template data<T>();
          *intercepts = intercepts_input.template data<T>();
          num_bounds = bounds_input.numel();
          num_slopes = slopes_input.numel();
          num_intercepts = intercepts_input.numel();
        }
        InferNumFunctionsPerGroup(
            num_bounds, num_slopes, num_intercepts, num_func_per_group, num_group);
        */
    }
}
