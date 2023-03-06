crate::ix!();

impl MaxPoolFunctor<CPUContext> {

    #[inline] pub fn global_pooling_forward_f32nchw(&self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const std::array<int, 2> X_dims = {N * C, HxW};
      const std::array<int, 2> Y_dims = {N * C, 1};
      math::ReduceMax<float, CPUContext>(
          2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
      return true;
        */
    }
    
    #[inline] pub fn global_pooling_forward_f32nhwc(&self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
            math::Set<float, CPUContext>(
          N * C, std::numeric_limits<float>::lowest(), Y, context);
      const float* X_ptr = X;
      float* Y_ptr = Y;
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> X_arr(X_ptr, C, HxW);
        EigenVectorArrayMap<float> Y_arr(Y_ptr, C);
        for (int j = 0; j < HxW; ++j) {
          Y_arr = Y_arr.max(X_arr.col(j));
        }
        X_ptr += HxW * C;
        Y_ptr += C;
      }
      return true;
        */
    }

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
