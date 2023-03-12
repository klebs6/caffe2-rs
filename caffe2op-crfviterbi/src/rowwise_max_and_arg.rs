crate::ix!();

#[inline] pub fn rowwise_max_and_arg(
    mat:         *const f32,
    n:           i32,
    d:           i32,
    row_max:     *mut f32,
    arg_max:     *mut i32)  
{
    todo!();
    /*
        auto eigenMat = ConstEigenMatrixMap<float>(mat, D, N);
      for (auto i = 0; i < D; i++) {
        // eigenMat.row(i) is equivalent to column i in mat
        rowMax[i] = eigenMat.row(i).maxCoeff(argMax + i);
      }
    */
}

#[inline] pub fn colwise_max_and_arg(
    mat:     *const f32,
    n:       i32,
    d:       i32,
    col_max: *mut f32,
    arg_max: *mut i32)  {
    
    todo!();
    /*
        auto eigenMat = ConstEigenMatrixMap<float>(mat, D, N);
      for (auto i = 0; i < N; i++) {
        // eigenMat.col(i) is equivalent to row i in mat
        colMax[i] = eigenMat.col(i).maxCoeff(argMax + i);
      }
    */
}
