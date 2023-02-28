crate::ix!();

use crate::{
    EigenArrayMap,
    ConstEigenArrayMap,
    MaxPoolFunctor,
    GradientMakerBase,
    AveragePoolFunctor,
    CPUContext,
    StorageOrder,
    OperatorDef,
};

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

#[inline] pub fn run_average_pool_gradient1d<T, const kOrder: StorageOrder>(
    n:                 i32,
    c:                 i32,
    x_size:            i32,
    y_size:            i32,
    kernel:            i32,
    stride:            i32,
    pad:               i32,
    count_include_pad: bool,
    dy:                *const T,
    dx:                *mut T)  {

    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_size);
      const T* dY_ptr = dY;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_size, 1)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_size);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_size, 1)
            : EigenArrayMap<T>(dX_ptr, C, X_size);
        for (int y = 0; y < Y_size; ++y) {
          const int l = std::max(y * stride - pad, 0);
          const int r = std::min(y * stride - pad + kernel, X_size);
          const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
          ComputeAveragePoolGradient1D<T, kOrder>(l, r, y, scale, dY_arr, &dX_arr);
        }
        dY_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

#[inline] pub fn run_average_pool_gradient2D<T, const kOrder: StorageOrder>(
    n:                 i32,
    c:                 i32,
    x_H:               i32,
    x_W:               i32,
    y_H:               i32,
    y_W:               i32,
    kernel_h:          i32,
    kernel_w:          i32,
    stride_h:          i32,
    stride_w:          i32,
    pad_t:             i32,
    pad_l:             i32,
    count_include_pad: bool,
    dY:                *const T,
    dX:                *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_H * X_W;
      const int Y_HxW = Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
      const T* dY_ptr = dY;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_H)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_W, X_H)
            : EigenArrayMap<T>(dX_ptr, C, X_HxW);
        for (int h = 0; h < Y_H; ++h) {
          const int t = std::max(h * stride_h - pad_t, 0);
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
          for (int w = 0; w < Y_W; ++w) {
            const int l = std::max(w * stride_w - pad_l, 0);
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
            const int y = h * Y_W + w;
            const T scale = T(1) /
                static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                                 : (b - t) * (r - l));
            ComputeAveragePoolGradient2D<T, kOrder>(
                X_W, t, b, l, r, y, scale, dY_arr, &dX_arr);
          }
        }
        dY_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

#[inline] pub fn run_average_pool_gradient3D<T, const kOrder: StorageOrder>(
    n:                  i32,
    c:                  i32,
    x_D:                i32,
    x_H:                i32,
    x_W:                i32,
    y_D:                i32,
    y_H:                i32,
    y_W:                i32,
    kernel_d:           i32,
    kernel_h:           i32,
    kernel_w:           i32,
    stride_d:           i32,
    stride_h:           i32,
    stride_w:           i32,
    pad_p:              i32,
    pad_t:              i32,
    pad_l:              i32,
    count_include_pad:  bool,
    dY:                 *const T,
    dX:                 *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_D * X_H * X_W;
      const int Y_HxW = Y_D * Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
      const T* dY_ptr = dY;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_D * Y_H)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_W, X_D * X_H)
            : EigenArrayMap<T>(dX_ptr, C, X_HxW);
        for (int d = 0; d < Y_D; ++d) {
          const int p = std::max(d * stride_d - pad_p, 0);
          const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);
          for (int h = 0; h < Y_H; ++h) {
            const int t = std::max(h * stride_h - pad_t, 0);
            const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
            for (int w = 0; w < Y_W; ++w) {
              const int l = std::max(w * stride_w - pad_l, 0);
              const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
              const int y = d * Y_H * Y_W + h * Y_W + w;
              const T scale = T(1) /
                  static_cast<T>(count_include_pad ? kernel_d * kernel_h * kernel_w
                                                   : (a - p) * (b - t) * (r - l));
              ComputeAveragePoolGradient3D<T, kOrder>(
                  X_H, X_W, p, a, t, b, l, r, y, scale, dY_arr, &dX_arr);
            }
          }
        }
        dY_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

#[inline] pub fn compute_max_pool_gradient1d<T, const kOrder: StorageOrder>(
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<T>,
    x_arr:  &ConstEigenArrayMap<T>,
    y_arr:  &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_gradient_1df32nchw(
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        dX_arr->col(0).segment(l, r - l) +=
          (X_arr.col(0).segment(l, r - l) == Y_arr(y)).cast<float>() * dY_arr(y);
    */
}

#[inline] pub fn compute_max_pool_gradient_1df32nhwc(
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = l; i < r; ++i) {
        dX_arr->col(i) +=
            (X_arr.col(i) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
      }
    */
}

#[inline] pub fn compute_max_pool_gradient2d<T, const kOrder: StorageOrder>(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<T>,
    x_arr:  &ConstEigenArrayMap<T>,
    y_arr:  &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_gradient_2df32nchw(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        dX_arr->block(l, t, r - l, b - t) +=
          (X_arr.block(l, t, r - l, b - t) == Y_arr(y)).cast<float>() * dY_arr(y);
    */
}

#[inline] pub fn compute_max_pool_gradient_2df32nhwc(
    w:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = t; i < b; ++i) {
        for (int j = l; j < r; ++j) {
          const int x = i * W + j;
          dX_arr->col(x) +=
              (X_arr.col(x) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
        }
      }
    */
}

#[inline] pub fn compute_max_pool_gradient3d<T, const kOrder: StorageOrder>(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<T>,
    x_arr:  &ConstEigenArrayMap<T>,
    y_arr:  &ConstEigenArrayMap<T>,
    dx_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_max_pool_gradient_3df32nchw(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = p; i < a; ++i) {
        dX_arr->block(l, i * H + t, r - l, b - t) +=
            (X_arr.block(l, i * H + t, r - l, b - t) == Y_arr(y)).cast<float>() *
            dY_arr(y);
      }
    */
}

#[inline] pub fn compute_max_pool_gradient_3df32nhwc(
    h:      i32,
    w:      i32,
    p:      i32,
    a:      i32,
    t:      i32,
    b:      i32,
    l:      i32,
    r:      i32,
    y:      i32,
    dy_arr: &ConstEigenArrayMap<f32>,
    x_arr:  &ConstEigenArrayMap<f32>,
    y_arr:  &ConstEigenArrayMap<f32>,
    dx_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        for (int i = p; i < a; ++i) {
        for (int j = t; j < b; ++j) {
          for (int k = l; k < r; ++k) {
            const int x = i * H * W + j * W + k;
            dX_arr->col(x) +=
                (X_arr.col(x) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
          }
        }
      }
    */
}

#[inline] pub fn run_max_pool_gradient1D<T, const kOrder: StorageOrder>(
    n:        i32,
    c:        i32,
    x_size:   i32,
    y_size:   i32,
    kernel:   i32,
    stride:   i32,
    pad:      i32,
    dY:       *const T,
    x:        *const T,
    y:        *const T,
    dX:       *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_size);
      const T* dY_ptr = dY;
      const T* X_ptr = X;
      const T* Y_ptr = Y;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_size, 1)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_size);
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
            : ConstEigenArrayMap<T>(X_ptr, C, X_size);
        ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(Y_ptr, Y_size, 1)
            : ConstEigenArrayMap<T>(Y_ptr, C, Y_size);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_size, 1)
            : EigenArrayMap<T>(dX_ptr, C, X_size);
        for (int y = 0; y < Y_size; ++y) {
          const int l = std::max(y * stride - pad, 0);
          const int r = std::min(y * stride - pad + kernel, X_size);
          ComputeMaxPoolGradient1D<T, kOrder>(
              l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
        }
        dY_ptr += Y_stride;
        X_ptr += X_stride;
        Y_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

#[inline] pub fn run_max_pool_gradient2D<T, const kOrder: StorageOrder>(
    n:         i32,
    c:         i32,
    x_H:       i32,
    x_W:       i32,
    y_H:       i32,
    y_W:       i32,
    kernel_h:  i32,
    kernel_w:  i32,
    stride_h:  i32,
    stride_w:  i32,
    pad_t:     i32,
    pad_l:     i32,
    dY:        *const T,
    x:         *const T,
    y:         *const T,
    dX:        *mut T) {
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_H * X_W;
      const int Y_HxW = Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
      const T* dY_ptr = dY;
      const T* X_ptr = X;
      const T* Y_ptr = Y;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_H)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(Y_ptr, Y_W, Y_H)
            : ConstEigenArrayMap<T>(Y_ptr, C, Y_HxW);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_W, X_H)
            : EigenArrayMap<T>(dX_ptr, C, X_HxW);
        for (int h = 0; h < Y_H; ++h) {
          const int t = std::max(h * stride_h - pad_t, 0);
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
          for (int w = 0; w < Y_W; ++w) {
            const int l = std::max(w * stride_w - pad_l, 0);
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
            const int y = h * Y_W + w;
            ComputeMaxPoolGradient2D<T, kOrder>(
                X_W, t, b, l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
          }
        }
        dY_ptr += Y_stride;
        X_ptr += X_stride;
        Y_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

#[inline] pub fn run_max_pool_gradient3D<T, const kOrder: StorageOrder>(
    n:           i32,
    c:           i32,
    x_D:         i32,
    x_H:         i32,
    x_W:         i32,
    y_D:         i32,
    y_H:         i32,
    y_W:         i32,
    kernel_d:    i32,
    kernel_h:    i32,
    kernel_w:    i32,
    stride_d:    i32,
    stride_h:    i32,
    stride_w:    i32,
    pad_p:       i32,
    pad_t:       i32,
    pad_l:       i32,
    dY:          *const T,
    x:           *const T,
    y:           *const T,
    dX:          *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_D * X_H * X_W;
      const int Y_HxW = Y_D * Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
      const T* dY_ptr = dY;
      const T* X_ptr = X;
      const T* Y_ptr = Y;
      T* dX_ptr = dX;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_D * Y_H)
            : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
            : ConstEigenArrayMap<T>(Y_ptr, C, Y_HxW);
        EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(dX_ptr, X_W, X_D * X_H)
            : EigenArrayMap<T>(dX_ptr, C, X_HxW);
        for (int d = 0; d < Y_D; ++d) {
          const int p = std::max(d * stride_d - pad_p, 0);
          const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);
          for (int h = 0; h < Y_H; ++h) {
            const int t = std::max(h * stride_h - pad_t, 0);
            const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
            for (int w = 0; w < Y_W; ++w) {
              const int l = std::max(w * stride_w - pad_l, 0);
              const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
              const int y = d * Y_H * Y_W + h * Y_W + w;
              ComputeMaxPoolGradient3D<T, kOrder>(
                  X_H, X_W, p, a, t, b, l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
            }
          }
        }
        dY_ptr += Y_stride;
        X_ptr += X_stride;
        Y_ptr += Y_stride;
        dX_ptr += X_stride;
      }
    */
}

impl AveragePoolFunctor<CPUContext> {
    
    #[inline] pub fn global_pooling_backward_f32nchw(&self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const f32,
        x:       *const f32,
        y:       *const f32,
        dx:      *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
          const int NxC = N * C;
          EigenArrayMap<float> dX_arr(dX, HxW, NxC);
          const float scale = 1.0f / static_cast<float>(HxW);
          for (int i = 0; i < NxC; ++i) {
            dX_arr.col(i).setConstant(dY[i] * scale);
          }
          return true;
        */
    }
    
    #[inline] pub fn global_pooling_backward_f32nhwc(&self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const f32,
        x:       *const f32,
        y:       *const f32,
        dx:      *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            ConstEigenArrayMap<float> dY_arr(dY, C, N);
      const float scale = 1.0f / static_cast<float>(HxW);
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float>(dX + i * HxW * C, C, HxW).colwise() =
            dY_arr.col(i) * scale;
      }
      return true;
        */
    }
}

impl AveragePoolFunctor<CPUContext> {
    
    #[inline] pub fn backward_cpu<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        dy:       *const T,
        x:        *const T,
        y:        *const T,
        dx:       *mut T,
        context:  *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int ndim = X_dims.size();
      switch (ndim) {
        case 1: {
          RunAveragePoolGradient1D<T, kOrder>(
              N,
              C,
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              count_include_pad,
              dY,
              dX);
          return true;
        }
        case 2: {
          RunAveragePoolGradient2D<T, kOrder>(
              N,
              C,
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              count_include_pad,
              dY,
              dX);
          return true;
        }
        case 3: {
          RunAveragePoolGradient3D<T, kOrder>(
              N,
              C,
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              count_include_pad,
              dY,
              dX);
          return true;
        }
        default: {
          CAFFE_THROW("Unsupported pooling dim: ", ndim);
          return false;
        }
      }
        */
    }
}

impl MaxPoolFunctor<CPUContext> {

    #[inline] pub fn global_pooling_backward_f32nchw(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const f32,
        x:       *const f32,
        y:       *const f32,
        dx:      *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const int NxC = N * C;
      ConstEigenArrayMap<float> X_arr(X, HxW, NxC);
      EigenArrayMap<float> dX_arr(dX, HxW, NxC);
      for (int i = 0; i < NxC; ++i) {
        dX_arr.col(i) = (X_arr.col(i) == Y[i]).template cast<float>() * dY[i];
      }
      return true;
        */
    }
    
    #[inline] pub fn global_pooling_backward_f32nhwc(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const f32,
        x:       *const f32,
        y:       *const f32,
        dx:      *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            ConstEigenArrayMap<float> Y_arr(Y, C, N);
      ConstEigenArrayMap<float> dY_arr(dY, C, N);
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> X_arr(X + i * HxW * C, C, HxW);
        EigenArrayMap<float> dX_arr(dX + i * HxW * C, C, HxW);
        for (int j = 0; j < HxW; ++j) {
          dX_arr.col(j) =
              (X_arr.col(j) == Y_arr.col(i)).template cast<float>() * dY_arr.col(i);
        }
      }
      return true;
        */
    }
    
    #[inline] pub fn backward_cpu<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        dy:       *const T,
        x:        *const T,
        y:        *const T,
        dx:       *mut T,
        context:  *mut CPUContext) -> bool {

        todo!();
        /*
            const int ndim = X_dims.size();
      switch (ndim) {
        case 1: {
          RunMaxPoolGradient1D<T, kOrder>(
              N,
              C,
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              dY,
              X,
              Y,
              dX);
          return true;
        }
        case 2: {
          RunMaxPoolGradient2D<T, kOrder>(
              N,
              C,
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              dY,
              X,
              Y,
              dX);
          return true;
        }
        case 3: {
          RunMaxPoolGradient3D<T, kOrder>(
              N,
              C,
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              dY,
              X,
              Y,
              dX);
          return true;
        }
        default: {
          CAFFE_THROW("Unsupported pooling dim: ", ndim);
          return false;
        }
      }
        */
    }
}

///-----------------------------
register_cpu_operator!{
    AveragePoolGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePoolGradient, 3}

num_outputs!{AveragePoolGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool1DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool1DGradient, 3}

num_outputs!{AveragePool1DGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool2DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool2DGradient, 3}

num_outputs!{AveragePool2DGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool3DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool3DGradient, 3}

num_outputs!{AveragePool3DGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPoolGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}
num_inputs!{MaxPoolGradient, 3}

num_outputs!{MaxPoolGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool1DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}
num_inputs!{MaxPool1DGradient, 3}

num_outputs!{MaxPool1DGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool2DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}
num_inputs!{MaxPool2DGradient, 3}

num_outputs!{MaxPool2DGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool3DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}
num_inputs!{MaxPool3DGradient, 3}

num_outputs!{MaxPool3DGradient, 1}

pub struct GetPoolGradient;

impl GetGradientDefs for GetPoolGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<std::string>{I(0), O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{AveragePool, GetPoolGradient}
register_gradient!{AveragePool1D, GetPoolGradient}
register_gradient!{AveragePool2D, GetPoolGradient}
register_gradient!{AveragePool3D, GetPoolGradient}

register_gradient!{MaxPool, GetPoolGradient}
register_gradient!{MaxPool1D, GetPoolGradient}
register_gradient!{MaxPool2D, GetPoolGradient}
register_gradient!{MaxPool3D, GetPoolGradient}
