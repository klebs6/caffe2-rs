crate::ix!();

impl MaxReducer<CPUContext> {

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
            ComputeReduceMinMaxGradient(
          dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data);
      return true;
        */
    }
}
