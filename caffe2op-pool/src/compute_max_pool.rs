crate::ix!();

#[inline] pub fn compute_max_pool1d<T, const kOrder: StorageOrder>(
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_1df32nchw(
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = X_arr.col(0).segment(l, r - l).maxCoeff();
    */
}

#[inline] pub fn compute_max_pool_1df32nhwc(
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y) = X_arr.col(l);
      for (int i = l + 1; i < r; ++i) {
        Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i));
      }
    */
}

#[inline] pub fn compute_max_pool2d<T, const kOrder: StorageOrder>(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_2d_f32_nchw(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = X_arr.block(l, t, r - l, b - t).maxCoeff();
    */
}

#[inline] pub fn compute_max_pool_2df32nhwc(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y).setConstant(std::numeric_limits<float>::lowest());
      for (int i = t; i < b; ++i) {
        for (int j = l; j < r; ++j) {
          Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i * W + j));
        }
      }
    */
}

#[inline] pub fn compute_max_pool3d<T, const kOrder: StorageOrder>(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_3df32nchw(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = std::numeric_limits<float>::lowest();
      for (int i = p; i < a; ++i) {
        (*Y_arr)(y) = std::max(
            (*Y_arr)(y), X_arr.block(l, i * H + t, r - l, b - t).maxCoeff());
      }
    */
}

#[inline] pub fn compute_max_pool_3df32nhwc(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y).setConstant(std::numeric_limits<float>::lowest());
      for (int i = p; i < a; ++i) {
        for (int j = t; j < b; ++j) {
          for (int k = l; k < r; ++k) {
            Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i * H * W + j * W + k));
          }
        }
      }
    */
}
