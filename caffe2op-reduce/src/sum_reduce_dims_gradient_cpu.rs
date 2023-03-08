crate::ix!();

// ReduceFrontMeanGradient
impl SumReduceDimsGradientOp<CPUContext, true, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {

        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr) {
          dXdata[i] = dYdata[col] / rows;
        } else if (row < lengths_data[col]) {
          dXdata[i] = dYdata[col] / lengths_data[col];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

impl SumReduceDimsGradientOp<CPUContext, false, true> {

    // ReduceBackMeanGradient
    #[inline] pub fn compute_on_cpu<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr) {
          dXdata[i] = dYdata[row] / cols;
        } else if (col < lengths_data[row]) {
          dXdata[i] = dYdata[row] / lengths_data[row];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

/// ReduceFrontSumGradient
impl SumReduceDimsGradientOp<CPUContext, true, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || row < lengths_data[col]) {
          dXdata[i] = dYdata[col];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}

// ReduceBackSumGradient
impl SumReduceDimsGradientOp<CPUContext, false, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const T,
        lengths_data: *const i32,
        d_xdata:      *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || col < lengths_data[row]) {
          dXdata[i] = dYdata[row];
        } else {
          dXdata[i] = 0;
        }
      }
        */
    }
}
