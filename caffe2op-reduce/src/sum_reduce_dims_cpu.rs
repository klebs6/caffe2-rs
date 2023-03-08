crate::ix!();

/// ReduceFrontMean: columnwise mean
impl SumReduceDimsOp<CPUContext, true, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {

        todo!();
        /*
            for (int j = 0; j < cols; j++) {
        T sum = in_data[j];
        int length = lengths_data == nullptr ? rows : lengths_data[j];
        for (int i = 1; i < length; i++) {
          sum += in_data[i * cols + j];
        }
        out_data[j] = sum / length;
      }
        */
    }
}

// ReduceBackMean: rowwise mean
impl SumReduceDimsOp<CPUContext, false, true> {

    #[inline] pub fn compute_on_cpu<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {

        todo!();
        /*
            for (int i = 0; i < rows; i++) {
        int offset = i * cols;
        T sum = in_data[offset];
        int length = lengths_data == nullptr ? cols : lengths_data[i];
        for (int j = 1; j < length; j++) {
          sum += in_data[offset + j];
        }
        out_data[i] = sum / length;
      }
        */
    }
}

// ReduceFrontSum: columnwise sum
impl SumReduceDimsOp<CPUContext, true, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {
    
        todo!();
        /*
            for (int j = 0; j < cols; j++) {
        T sum = in_data[j];
        int length = lengths_data == nullptr ? rows : lengths_data[j];
        for (int i = 1; i < length; i++) {
          sum += in_data[i * cols + j];
        }
        out_data[j] = sum;
      }
        */
    }
}

// ReduceBackSum: rowwise sum
impl SumReduceDimsOp<CPUContext, false, false> {

    #[inline] pub fn compute<T>(&mut self, 
        rows:         i32,
        cols:         i32,
        in_data:      *const T,
        lengths_data: *const i32,
        out_data:     *mut T)  {
    
        todo!();
        /*
            for (int i = 0; i < rows; i++) {
        int offset = i * cols;
        T sum = in_data[offset];
        int length = lengths_data == nullptr ? cols : lengths_data[i];
        for (int j = 1; j < length; j++) {
          sum += in_data[offset + j];
        }
        out_data[i] = sum;
      }
        */
    }
}
