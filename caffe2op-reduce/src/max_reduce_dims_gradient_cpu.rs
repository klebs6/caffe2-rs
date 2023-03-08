crate::ix!();

// ReduceFrontMaxGradient
impl MaxReduceDimsGradientOp<f32, CPUContext, true> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const f32,
        xdata:        *const f32,
        ydata:        *const f32,
        lengths_data: *const i32,
        d_xdata:      *mut f32)  {
        
        todo!();
        /*
            int len = cols * rows;
      for (int i = 0; i < len; i++) {
        int col = i % cols;
        int row = i / cols;
        if (lengths_data != nullptr && row >= lengths_data[col]) {
          dXdata[i] = 0.0f;
        } else {
          dXdata[i] = Xdata[i] == Ydata[col] ? dYdata[col] : 0.0f;
        }
      }
        */
    }
}

// ReduceBackMaxGradient
impl MaxReduceDimsGradientOp<f32, CPUContext, false> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const f32,
        xdata:        *const f32,
        ydata:        *const f32,
        lengths_data: *const i32,
        d_xdata:      *mut f32)  {

        todo!();
        /*
            int len = cols * rows;
      for (int i = 0; i < len; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || col < lengths_data[row]) {
          dXdata[i] = Xdata[i] == Ydata[row] ? dYdata[row] : 0.0f;
        } else {
          dXdata[i] = 0.0f;
        }
      }
        */
    }
}
