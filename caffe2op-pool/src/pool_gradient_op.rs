crate::ix!();

///----------------------------------
#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct PoolGradientOp<T, Context, Functor> {
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
