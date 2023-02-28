crate::ix!();

pub struct PoolOp<T, Context, Functor> {
    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    functor: Functor,
    phantom: PhantomData<T>,
}

impl<T, Context, Functor> PoolOp<T, Context, Functor> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...), functor_(*this) 

        const int kernel_size = kernel_.size();
        for (int i = 0; i < kernel_size; ++i) {
          CAFFE_ENFORCE_EQ(
              dilation_[i], 1, "Pooling op does not support dilation right now.");
        }
        if (!global_pooling_) {
          for (int i = 0; i < kernel_size; ++i) {
            CAFFE_ENFORCE(
                pads_[i] < kernel_[i] && pads_[i + kernel_size] < kernel_[i],
                "Pad should be smaller than kernel.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        auto* Y = Output(0);
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
        const T* X_data = X.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        if (N == 0) {
          return true;
        }
        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          return functor_.template GlobalPoolingForward<T, StorageOrder::NCHW>(
              N, C, HxW, X_data, Y_data, &context_);
        }
        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(*Y);
        return functor_.template Forward<T, StorageOrder::NCHW>(
            N,
            C,
            X_HW_dims,
            Y_HW_dims,
            kernel_,
            dilation_,
            stride_,
            pads_,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        auto* Y = Output(0);
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = X.dim32(ndim - 1);
        ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
        const T* X_data = X.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        if (N == 0) {
          return true;
        }
        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          return functor_.template GlobalPoolingForward<T, StorageOrder::NHWC>(
              N, C, HxW, X_data, Y_data, &context_);
        }
        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(*Y);
        return functor_.template Forward<T, StorageOrder::NHWC>(
            N,
            C,
            X_HW_dims,
            Y_HW_dims,
            kernel_,
            dilation_,
            stride_,
            pads_,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
}



///----------------------------------
pub struct PoolGradientOp<T, Context, Functor> {
    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base:    ConvPoolOpBase<Context>,

    functor: Functor,
    phantom: PhantomData<T>,
}

impl<T,Context,Functor> PoolGradientOp<T,Context,Functor> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...), functor_(*this)
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& Y = Input(1);
        const auto& dY = Input(2);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(Y);
        ConvPoolOpBase<Context>::ComputePads(X_HW_dims);
        const T* dY_data = dY.template data<T>();
        const T* X_data = X.template data<T>();
        const T* Y_data = Y.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        if (N == 0) {
          return true;
        }
        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          return functor_.template GlobalPoolingBackward<T, StorageOrder::NCHW>(
              N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
        }
        return functor_.template Backward<T, StorageOrder::NCHW>(
            N,
            C,
            X_HW_dims,
            Y_HW_dims,
            kernel_,
            dilation_,
            stride_,
            pads_,
            dY_data,
            X_data,
            Y_data,
            dX_data,
            &context_);
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& Y = Input(1);
        const auto& dY = Input(2);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = X.dim32(ndim - 1);
        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(Y);
        ConvPoolOpBase<Context>::ComputePads(X_HW_dims);
        const T* dY_data = dY.template data<T>();
        const T* X_data = X.template data<T>();
        const T* Y_data = Y.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        if (N == 0) {
          return true;
        }
        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          return functor_.template GlobalPoolingBackward<T, StorageOrder::NHWC>(
              N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
        }
        return functor_.template Backward<T, StorageOrder::NHWC>(
            N,
            C,
            X_HW_dims,
            Y_HW_dims,
            kernel_,
            dilation_,
            stride_,
            pads_,
            dY_data,
            X_data,
            Y_data,
            dX_data,
            &context_);
        */
    }
}

///-------------------------------------------------

#[test] fn average_pool_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "AveragePool",
        ["X"],
        ["Y"],
        kernel=2,
        stride=2,
    )

    workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
    print("X:\n", workspace.FetchBlob("X"), "\n")
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[[[-0.2883434   0.43498734  0.05417408  1.912558    0.09390241
        -0.33173105]
       [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
         0.9989975 ]
       [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
         0.33195344]
       [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
         0.2586746 ]
       [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
        -0.97614396]
       [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
        -0.37085298]]]]

    Y:
     [[[[0.7462672  0.83399826 0.2948959 ]
       [0.4843537  0.3506009  0.35500962]
       [0.9251013  0.19026303 0.13366827]]]]
    */
}

/**
  | consumes an input blob and applies average
  | pooling across the the blob according
  | to kernel sizes, stride sizes, pad lengths
  | and dilation. Average pooling consists
  | of taking the average value of a subset
  | of the input tensor according to the
  | kernel size and downsampling the data
  | into the output blob for further processing.
  | The `brew` module has a wrapper for this
  | operator for use in a `ModelHelper`
  | object.
  | 
  | Pooling layers reduce the spatial dimensionality
  | of the input blob. Each of the output
  | blob's dimensions will reduce according
  | to:
  | 
  | $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
  |
  */
pub struct AveragePoolFunctor<Context> {
    count_include_pad: bool,

    /// {Context::GetDeviceType()};
    ones: Tensor,

    phantom: PhantomData<Context>,

}

impl<Context> AveragePoolFunctor<Context> {
    
    pub fn new(op: &OperatorStorage) -> Self {
        todo!();
        /*
            : count_include_pad( op.template GetSingleArgument<bool>("count_include_pad", false))
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        x:        *const T,
        y:        *mut T,
        context:  *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const T,
        x:       *const T,
        y:       *const T,
        dx:      *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
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
        context:  *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

///------------------------------
#[test] fn max_pool_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "MaxPool",
        ["X"],
        ["Y"],
        kernel=2,
        stride=2,
    )

    workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
    print("X:\n", workspace.FetchBlob("X"), "\n")
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04  1.1088650e+00
        -2.1476576e+00 -3.5070452e-01]
       [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
         1.2798511e+00 -3.2214901e-01]
       [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
        -9.6424848e-02 -3.9696163e-01]
       [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
        -9.1898698e-01 -1.9609968e-01]
       [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
        -8.1293899e-01 -7.8803545e-01]
       [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
         7.2035044e-01  2.7642491e-01]]]]

    Y:
     [[[[-0.28534958  1.108865    1.2798511 ]
       [ 1.6845188  -0.266332   -0.09642485]
       [ 2.4605224   0.13020347  0.72035044]]]]
    */
}

/**
  | consumes an input blob and applies max
  | pooling across the the blob according
  | to kernel sizes, stride sizes, pad lengths
  | and dilation. Max pooling consists
  | of taking the maximum value of a subset
  | of the input tensor according to the
  | kernel size and downsampling the data
  | into the output blob for further processing.
  | The `brew` module has a wrapper for this
  | operator for use in a `ModelHelper`
  | object.
  | 
  | Pooling layers reduce the spatial dimensionality
  | of the input blob. Each of the output
  | blob's dimensions will reduce according
  | to:
  | 
  | $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
  |
  */
pub struct MaxPoolFunctor<Context> {
    

    phantom: PhantomData<Context>,

}

impl<Context> MaxPoolFunctor<Context> {
    
    pub fn new(op: &OperatorStorage) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        x:        *const T,
        y:        *mut T,
        context:  *mut Context) -> bool {

        todo!();
        /*

        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        dy:      *const T,
        x:       *const T,
        y:       *const T,
        dx:      *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
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
        context:  *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

#[inline] pub fn compute_average_pool1d<T, const kOrder: StorageOrder>(
    l:     i32,
    r:     i32,
    y:     i32,
    scale: T,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_1df32nchw(
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = X_arr.col(0).segment(l, r - l).sum() * scale;
    */
}

#[inline] pub fn compute_average_pool_1df32nhwc(
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y) = X_arr.col(l);
      for (int i = l + 1; i < r; ++i) {
        Y_arr->col(y) += X_arr.col(i);
      }
      Y_arr->col(y) *= scale;
    */
}

#[inline] pub fn compute_average_pool2d<T, const kOrder: StorageOrder>(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: T,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_2d_f32_nchw(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = X_arr.block(l, t, r - l, b - t).sum() * scale;
    */
}

#[inline] pub fn compute_average_pool_2df32nhwc(
    w:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y).setZero();
      for (int i = t; i < b; ++i) {
        for (int j = l; j < r; ++j) {
          Y_arr->col(y) += X_arr.col(i * W + j);
        }
      }
      Y_arr->col(y) *= scale;
    */
}

#[inline] pub fn compute_average_pool3d<T, const kOrder: StorageOrder>(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: T,
    x_arr: &ConstEigenArrayMap<T>,
    y_arr: *mut EigenArrayMap<T>)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn compute_average_pool_3df32nchw(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        (*Y_arr)(y) = 0;
      for (int i = p; i < a; ++i) {
        (*Y_arr)(y) += X_arr.block(l, i * H + t, r - l, b - t).sum();
      }
      (*Y_arr)(y) *= scale;
    */
}

#[inline] pub fn compute_average_pool_3df32nhwc(
    h:     i32,
    w:     i32,
    p:     i32,
    a:     i32,
    t:     i32,
    b:     i32,
    l:     i32,
    r:     i32,
    y:     i32,
    scale: f32,
    x_arr: &ConstEigenArrayMap<f32>,
    y_arr: *mut EigenArrayMap<f32>)  {
    
    todo!();
    /*
        Y_arr->col(y).setZero();
      for (int i = p; i < a; ++i) {
        for (int j = t; j < b; ++j) {
          for (int k = l; k < r; ++k) {
            Y_arr->col(y) += X_arr.col(i * H * W + j * W + k);
          }
        }
      }
      Y_arr->col(y) *= scale;
    */
}

#[inline] pub fn run_average_pool1D<T, StorageOrder>(
    n:                  i32,
    c:                  i32,
    x_size:             i32,
    y_size:             i32,
    kernel:             i32,
    stride:             i32,
    pad:                i32,
    count_include_pad:  bool,
    x:                  *const T,
    y:                  *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
            : ConstEigenArrayMap<T>(X_ptr, C, X_size);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_size, 1)
            : EigenArrayMap<T>(Y_ptr, C, Y_size);
        for (int y = 0; y < Y_size; ++y) {
          const int l = std::max(y * stride - pad, 0);
          const int r = std::min(y * stride - pad + kernel, X_size);
          const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
          ComputeAveragePool1D<T, kOrder>(l, r, y, scale, X_arr, &Y_arr);
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}

#[inline] pub fn run_average_pool2D<T, const kOrder: StorageOrder>(
    n:                  i32,
    c:                  i32,
    x_H:                i32,
    x_W:                i32,
    y_H:                i32,
    y_W:                i32,
    kernel_h:           i32,
    kernel_w:           i32,
    stride_h:           i32,
    stride_w:           i32,
    pad_t:              i32,
    pad_l:              i32,
    count_include_pad:  bool,
    x:                  *const T,
    y:                  *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_H * X_W;
      const int Y_HxW = Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_W, Y_H)
            : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
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
            ComputeAveragePool2D<T, kOrder>(
                X_W, t, b, l, r, y, scale, X_arr, &Y_arr);
          }
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}


#[inline] pub fn run_average_pool3D<T, const kOrder: StorageOrder>(
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
    x:                  *const T,
    y:                  *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_D * X_H * X_W;
      const int Y_HxW = Y_D * Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
            : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
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
              ComputeAveragePool3D<T, kOrder>(
                  X_H, X_W, p, a, t, b, l, r, y, scale, X_arr, &Y_arr);
            }
          }
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}

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

#[inline] pub fn run_max_pool1D<T, const kOrder: StorageOrder>(
    n:        i32,
    c:        i32,
    x_size:   i32,
    y_size:   i32,
    kernel:   i32,
    stride:   i32,
    pad:      i32,
    x:        *const T,
    y:        *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
            : ConstEigenArrayMap<T>(X_ptr, C, X_size);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_size, 1)
            : EigenArrayMap<T>(Y_ptr, C, Y_size);
        for (int y = 0; y < Y_size; ++y) {
          const int l = std::max(y * stride - pad, 0);
          const int r = std::min(y * stride - pad + kernel, X_size);
          ComputeMaxPool1D<T, kOrder>(l, r, y, X_arr, &Y_arr);
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}

#[inline] pub fn run_max_pool2D<T, const kOrder: StorageOrder>(
    n:           i32,
    c:           i32,
    x_H:         i32,
    x_W:         i32,
    y_H:         i32,
    y_W:         i32,
    kernel_h:    i32,
    kernel_w:    i32,
    stride_h:    i32,
    stride_w:    i32,
    pad_t:       i32,
    pad_l:       i32,
    x:           *const T,
    y:           *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_H * X_W;
      const int Y_HxW = Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_W, Y_H)
            : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
        for (int h = 0; h < Y_H; ++h) {
          const int t = std::max(h * stride_h - pad_t, 0);
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
          for (int w = 0; w < Y_W; ++w) {
            const int l = std::max(w * stride_w - pad_l, 0);
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
            const int y = h * Y_W + w;
            ComputeMaxPool2D<T, kOrder>(X_W, t, b, l, r, y, X_arr, &Y_arr);
          }
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}

#[inline] pub fn run_max_pool3D<T, const kOrder: StorageOrder>(
    n:         i32,
    c:         i32,
    x_D:       i32,
    x_H:       i32,
    x_W:       i32,
    y_D:       i32,
    y_H:       i32,
    y_W:       i32,
    kernel_d:  i32,
    kernel_h:  i32,
    kernel_w:  i32,
    stride_d:  i32,
    stride_h:  i32,
    stride_w:  i32,
    pad_p:     i32,
    pad_t:     i32,
    pad_l:     i32,
    x:         *const T,
    y:         *mut T) 
{
    todo!();
    /*
        const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
      const int X_HxW = X_D * X_H * X_W;
      const int Y_HxW = Y_D * Y_H * Y_W;
      const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
      const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
      const T* X_ptr = X;
      T* Y_ptr = Y;
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
            ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
            : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
        EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
            ? EigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
            : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
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
              ComputeMaxPool3D<T, kOrder>(
                  X_H, X_W, p, a, t, b, l, r, y, X_arr, &Y_arr);
            }
          }
        }
        X_ptr += X_stride;
        Y_ptr += Y_stride;
      }
    */
}

impl AveragePoolFunctor<CPUContext> {

    #[inline] pub fn global_pooling_forward_f32nhwc(&self, 
        n:       i32,
        c:       i32,
        hxw:     i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
            math::Set<float, CPUContext>(N * C, 0.0f, Y, context);
      const float* X_ptr = X;
      float* Y_ptr = Y;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < HxW; ++j) {
          math::Add<float, CPUContext>(C, Y_ptr, X_ptr + j * C, Y_ptr, context);
        }
        X_ptr += HxW * C;
        Y_ptr += C;
      }
      math::Scale<float, float, CPUContext>(
          N * C, 1.0f / static_cast<float>(HxW), Y, Y, context);
      return true;
        */
    }
    
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
      math::ReduceMean<float, CPUContext>(
          2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
      return true;
        */
    }
}

#[macro_export] macro_rules! caffe2_specialized_average_pool_functor_forward {
    ($T:ident, $($kOrder:ident)::+) => {
        /*
        
          template <>                                                                
          template <>                                                                
          bool AveragePoolFunctor<CPUContext>::Forward<T, kOrder>(                   
              const int N,                                                           
              const int C,                                                           
              const std::vector<int>& X_dims,                                        
              const std::vector<int>& Y_dims,                                        
              const std::vector<int>& kernel,                                        
              const std::vector<int>& dilation,                                      
              const std::vector<int>& stride,                                        
              const std::vector<int>& pads,                                          
              const T* X,                                                            
              T* Y,                                                                  
              CPUContext* /* context */) const {                                     
            const int ndim = X_dims.size();                                          
            switch (ndim) {                                                          
              case 1: {                                                              
                RunAveragePool1D<T, kOrder>(                                         
                    N,                                                               
                    C,                                                               
                    X_dims[0],                                                       
                    Y_dims[0],                                                       
                    kernel[0],                                                       
                    stride[0],                                                       
                    pads[0],                                                         
                    count_include_pad,                                               
                    X,                                                               
                    Y);                                                              
                return true;                                                         
              }                                                                      
              case 2: {                                                              
                if (std::is_same<T, float>::value && kOrder == StorageOrder::NCHW && 
                    pool_op_util::IsNeon4x4p0s0Eligible(                             
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
                        pads[2],                                                     
                        pads[3],                                                     
                        dilation[0],                                                 
                        dilation[1],                                                 
                        X,                                                           
                        Y)) {                                                        
                  pool_op_util::RunNeonAveragePool4x4p0s0NCHW(                       
                      N, C, X_dims[0], X_dims[1], X, Y);                             
                } else {                                                             
                  RunAveragePool2D<T, kOrder>(                                       
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
                      X,                                                             
                      Y);                                                            
                }                                                                    
                return true;                                                         
              }                                                                      
              case 3: {                                                              
                RunAveragePool3D<T, kOrder>(                                         
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
                    X,                                                               
                    Y);                                                              
                return true;                                                         
              }                                                                      
              default: {                                                             
                CAFFE_THROW("Unsupported pooling dim: ", ndim);                      
                return false;                                                        
              }                                                                      
            }                                                                        
          }
        */
    }
}

caffe2_specialized_average_pool_functor_forward!{f32, StorageOrder::NCHW}
caffe2_specialized_average_pool_functor_forward!{f32, StorageOrder::NHWC}

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
}

#[macro_export] macro_rules! caffe2_specialized_max_pool_functor_forward {
    ($T:ident, $($kOrder:ident)::+) => {
        /*
        
          template <>                                                                 
          template <>                                                                 
          bool MaxPoolFunctor<CPUContext>::Forward<T, kOrder>(                        
              const int N,                                                            
              const int C,                                                            
              const std::vector<int>& X_dims,                                         
              const std::vector<int>& Y_dims,                                         
              const std::vector<int>& kernel,                                         
              const std::vector<int>& dilation,                                       
              const std::vector<int>& stride,                                         
              const std::vector<int>& pads,                                           
              const T* X,                                                             
              T* Y,                                                                   
              CPUContext* /* context */) const {                                      
            const int ndim = X_dims.size();                                           
            switch (ndim) {                                                           
              case 1: {                                                               
                RunMaxPool1D<T, kOrder>(                                              
                    N, C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y); 
                return true;                                                          
              }                                                                       
              case 2: {                                                               
                if (std::is_same<T, float>::value && kOrder == StorageOrder::NCHW &&  
                    pool_op_util::IsNeon2x2p0s0Eligible(                              
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
                        pads[2],                                                      
                        pads[3],                                                      
                        dilation[0],                                                  
                        dilation[1],                                                  
                        X,                                                            
                        Y)) {                                                         
                  pool_op_util::RunNeonMaxPool2x2p0s0NCHW(                            
                      N, C, X_dims[0], X_dims[1], X, Y);                              
                } else {                                                              
                  RunMaxPool2D<T, kOrder>(                                            
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
                      X,                                                              
                      Y);                                                             
                }                                                                     
                return true;                                                          
              }                                                                       
              case 3: {                                                               
                RunMaxPool3D<T, kOrder>(                                              
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
                    X,                                                                
                    Y);                                                               
                return true;                                                          
              }                                                                       
              default: {                                                              
                CAFFE_THROW("Unsupported pooling dim: ", ndim);                       
                return false;                                                         
              }                                                                       
            }                                                                         
          }
        */
    }
}

caffe2_specialized_max_pool_functor_forward!{f32, StorageOrder::NCHW}
caffe2_specialized_max_pool_functor_forward!{f32, StorageOrder::NHWC}

#[inline] pub fn average_pool_doc_generator(dim: *const u8) -> fn(_u0: &mut OpSchema) -> c_void {
    
    todo!();
    /*
        return [=](OpSchema& schema) {
        std::string doc = "AveragePool{dim} {pool_doc}";
        c10::ReplaceAll(doc, "{dim}", dim);
        c10::ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
        schema.SetDoc(doc);
        schema.Input(
            0,
            "X",
            "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
        schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
        // schema.Arg(
        //     "kernel", "*(type: int)* Size of the window to take an average
        //     over.");
        // schema.Arg("stride", "*(type: int)* Stride of the window.");
        // schema.Arg(
        //     "pad",
        //     "*(type: int)* Implicit zero padding to be added on both sides.");
        // schema.Arg(
        //     "dilation",
        //     "*(type: int)* Parameter that controls the stride of elements in the
        //     " "window.");
        // schema.Arg(
        //     "order",
        //     "*(type: string; default: 'NCHW')* Order of the blob dimensions.");
        // schema.Arg(
        //     "count_include_pad",
        //     "*(type: bool; default: False)* When True, will include the "
        //     "zero-padding in the averaging.");
      };
    */
}

#[inline] pub fn max_pool_doc_generator(dim: *const u8) -> fn(_u0: &mut OpSchema) -> c_void {
    
    todo!();
    /*
        return [=](OpSchema& schema) {
        std::string doc = "MaxPool{dim} {pool_doc}";
        c10::ReplaceAll(doc, "{dim}", dim);
        c10::ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc);
        schema.SetDoc(doc);
        schema.Input(
            0,
            "X",
            "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
        schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
        /*
        schema.Arg("kernel", "*(type: int)* Size of the window to take an average
        over."); schema.Arg("stride", "*(type: int)* Stride of the window.");
        schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both
        sides."); schema.Arg("dilation", "*(type: int)* Parameter that controls
        the stride of elements in the window."); schema.Arg("order", "*(type:
        string; default: 'NCHW')* Order of the blob dimensions.");
        */
      };
    */
}

///-------------
register_cpu_operator!{
    AveragePool,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator(""))

num_inputs!{AveragePool, 1}

num_outputs!{AveragePool, 1}

tensor_inference_function!{AveragePool, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool}

///-------------
register_cpu_operator!{AveragePool1D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator("1D"))

num_inputs!{AveragePool1D, 1}

num_outputs!{AveragePool1D, 1}

tensor_inference_function!{AveragePool1D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool1D, "AveragePool"}


///-------------
register_cpu_operator!{
    AveragePool2D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
//.FillUsing(AveragePoolDocGenerator("2D"))

num_inputs!{AveragePool2D, 1}

num_outputs!{AveragePool2D, 1}

tensor_inference_function!{AveragePool2D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool2D, "AveragePool"}

///-------------
register_cpu_operator!{
    AveragePool3D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator("3D"))

num_inputs!{AveragePool3D, 1}

num_outputs!{AveragePool3D, 1}

tensor_inference_function!{
    AveragePool3D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool3D, "AveragePool"}

///-------------
register_cpu_operator!{
    MaxPool,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator(""))

num_inputs!{MaxPool, 1}

num_outputs!{MaxPool, 1}

tensor_inference_function!{MaxPool, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool}

///-------------
register_cpu_operator!{
    MaxPool1D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("1D"))

num_inputs!{MaxPool1D, 1}

num_outputs!{MaxPool1D, 1}

tensor_inference_function!{MaxPool1D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool1D, "MaxPool"}

///-------------
register_cpu_operator!{
    MaxPool2D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("2D"))

num_inputs!{MaxPool2D, 1}

num_outputs!{MaxPool2D, 1}

tensor_inference_function!{MaxPool2D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool2D, "MaxPool"}

///-------------
register_cpu_operator!{
    MaxPool3D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("3D"))

num_inputs!{MaxPool3D, 1}

num_outputs!{MaxPool3D, 1}

tensor_inference_function!{MaxPool3D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool3D, "MaxPool"}

