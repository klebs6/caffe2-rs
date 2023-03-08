crate::ix!();

// ReduceFrontMax
impl MaxReduceDimsOp<f32, CPUContext, true> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        data:         *const f32,
        lengths_data: *const i32,
        out_data:     *mut f32)  {

        todo!();
        /*
            for (int i = 0; i < cols; i++) {
        float mx = data[i];
        int length = lengths_data == nullptr ? rows : lengths_data[i];
        for (int j = 1; j < length; j++) {
          mx = std::max(mx, data[j * cols + i]);
        }
        out_data[i] = mx;
      }
        */
    }
}

// ReduceBackMax
impl MaxReduceDimsOp<f32, CPUContext, false> {
    
    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        data:         *const f32,
        lengths_data: *const i32,
        out_data:     *mut f32)  {

        todo!();
        /*
            for (int i = 0; i < rows; i++) {
        float mx = data[i * cols];
        int length = lengths_data == nullptr ? cols : lengths_data[i];
        for (int j = 1; j < length; j++) {
          mx = std::max(mx, data[i * cols + j]);
        }
        out_data[i] = mx;
      }
        */
    }
}
