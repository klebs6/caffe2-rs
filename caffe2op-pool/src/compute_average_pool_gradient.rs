crate::ix!();

#[inline] pub fn compute_average_pool_gradient1d<T, const kOrder: StorageOrder>(
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  T,
    dy_arr: &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_gradient_1df32nchw(
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        dX_arr->col(0).segment(l, r - l) += dY_arr(y) * scale;
    */
}

#[inline] pub fn compute_average_pool_gradient_1df32nhwc(
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = l; i < r; ++i) {
        dX_arr->col(i) += dY_arr.col(y) * scale;
      }
    */
}

#[inline] pub fn compute_average_pool_gradient2d<T, const kOrder: StorageOrder>(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  T,
    dy_arr: &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_gradient_2df32nchw(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        dX_arr->block(l, t, r - l, b - t) += dY_arr(y) * scale;
    */
}

#[inline] pub fn compute_average_pool_gradient_2df32nhwc(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = t; i < b; ++i) {
        for (int j = l; j < r; ++j) {
          dX_arr->col(i * W + j) += dY_arr.col(y) * scale;
        }
      }
    */
}

#[inline] pub fn compute_average_pool_gradient3d<T, const kOrder: StorageOrder>(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  T,
    dy_arr: &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_gradient_3df32nchw(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = p; i < a; ++i) {
        dX_arr->block(l, i * H + t, r - l, b - t) += dY_arr(y) * scale;
      }
    */
}

#[inline] pub fn compute_average_pool_gradient_3df32nhwc(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    scale:  f32,
    dy_arr: &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = p; i < a; ++i) {
        for (int j = t; j < b; ++j) {
          for (int k = l; k < r; ++k) {
            dX_arr->col(i * H * W + j * W + k) += dY_arr.col(y) * scale;
          }
        }
      }
    */
}
