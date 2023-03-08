crate::ix!();

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {

    /**
      | num_func_per_group is the number of pieces
      | of linear functions of each group.
      |
      | num_group: The number of groups of linear
      | functions. Each group is for transforming
      | one column of predictions.
      */
    #[inline] pub fn infer_num_functions_per_group(
        &mut self, 
        num_bounds:          i64,
        num_slopes:          i64,
        num_intercepts:      i64,
        num_func_per_group:  *mut i64,
        num_group:           *mut i64)  
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(num_slopes, num_intercepts);

        // This is based on the facts:
        // 1. in each group, the num of bounds minus the num of slopes is 1;
        // 2. each group has the same number of pieces.
        *num_group = num_bounds - num_slopes;
        CAFFE_ENFORCE_GT(*num_group, 0);
        if (binary_) {
          CAFFE_ENFORCE_EQ(*num_group, 1);
        }
        *num_func_per_group = num_slopes / *num_group;
        CAFFE_ENFORCE_GT(*num_func_per_group, 0);
        CAFFE_ENFORCE_EQ(num_slopes % *num_group, 0);
        */
    }
}
