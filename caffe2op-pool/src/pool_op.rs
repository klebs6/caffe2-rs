crate::ix!();

#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct PoolOp<T, Context, Functor> {
    base:    ConvPoolOpBase<Context>,
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
