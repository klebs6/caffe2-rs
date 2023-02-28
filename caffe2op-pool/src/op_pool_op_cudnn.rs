crate::ix!();

use crate::{
    AveragePoolFunctor,
    CUDAContext,
    ConvPoolOpBase,
    CudnnDataType,
    CudnnPoolingDescriptor,
    CudnnPoolingMode,
    CudnnTensorDescriptor,
    CudnnWrapper,
    MaxPoolFunctor,
    OperatorStorage,
    StorageOrder,
};

                                                                                                                                                                                                                                                                                                    
#[inline] pub fn set_tensor_descriptor(
    data_type: CudnnDataType,
    order:     StorageOrder,
    dims:      &Vec<i64>,
    desc:      *mut CudnnTensorDescriptor)  
{
    
    todo!();
    /*
        const int ndim = dims.size();
      const int N = dims[0];
      const int C = order == StorageOrder::NCHW ? dims[1] : dims[ndim - 1];
      switch (ndim) {
        case 4: {
          const int H = order == StorageOrder::NCHW ? dims[2] : dims[1];
          const int W = order == StorageOrder::NCHW ? dims[3] : dims[2];
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              *desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
          break;
        }
        case 5: {
          const int D = order == StorageOrder::NCHW ? dims[2] : dims[1];
          const int H = order == StorageOrder::NCHW ? dims[3] : dims[2];
          const int W = order == StorageOrder::NCHW ? dims[4] : dims[3];
          const std::array<int, 5> dims_arr = {N, C, D, H, W};
          const std::array<int, 5> strides_arr = order == StorageOrder::NCHW
              ? std::array<int, 5>{C * D * H * W, D * H * W, H * W, W, 1}
              : std::array<int, 5>{D * H * W * C, 1, H * W * C, W * C, C};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              *desc, data_type, 5, dims_arr.data(), strides_arr.data()));
          break;
        }
        default: {
          CAFFE_THROW("Unsupported tensor dim: ", ndim);
          break;
        }
      }
    */
}

///------------------------------
pub struct CudnnPoolOp<Functor> {
    base:            ConvPoolOpBase<CUDAContext>,
    cudnn_wrapper:   CudnnWrapper,
    x_desc:          CudnnTensorDescriptor,
    y_desc:          CudnnTensorDescriptor,
    pooling_desc:    CudnnPoolingDescriptor,
    functor:         Functor,
    equal_padding:   bool,
    cached_X_dims:   Vec<i64>,
}

impl<Functor> CudnnPoolOp<Functor> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            functor_(*this),
            equal_padding_(std::equal(
                pads_.cbegin(),
                pads_.cbegin() + kernel_.size(),
                pads_.cbegin() + kernel_.size())) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
        CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
        if (!global_pooling_ && equal_padding_) {
          if (kernel_.size() == 2) {
            CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
                pooling_desc_,
                functor_.GetPoolingMode(),
                CUDNN_NOT_PROPAGATE_NAN,
                kernel_h(),
                kernel_w(),
                pad_t(),
                pad_l(),
                stride_h(),
                stride_w()));
          } else if (kernel_.size() == 3) {
            CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
                pooling_desc_,
                functor_.GetPoolingMode(),
                CUDNN_NOT_PROPAGATE_NAN,
                kernel_.size(),
                kernel_.data(),
                pads_.data(),
                stride_.data()));
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, C);
        auto* Y = Output(0, sizes, at::dtype<T>());
        const T* X_data = X.template data<T>();
        T* Y_data = Y->template mutable_data<T>();

        if (N == 0) {
          return true;
        }

        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          if (order_ == StorageOrder::NCHW) {
            return functor_.template GlobalPoolingForward<T, StorageOrder::NCHW>(
                N, C, HxW, X_data, Y_data, &context_);
          } else {
            return functor_.template GlobalPoolingForward<T, StorageOrder::NHWC>(
                N, C, HxW, X_data, Y_data, &context_);
          }
        }

        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(*Y);
        if (order_ == StorageOrder::NHWC) {
          // Cudnn Pooling on NHWC order is very slow, fallback to CUDA
          // implementation.
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
        } else if (!equal_padding_ || ndim == 3) {
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
        }

        const std::vector<std::int64_t> X_dims = X.sizes().vec();
        const std::vector<std::int64_t> Y_dims = Y->sizes().vec();
        if (cached_X_dims_ != X_dims) {
          constexpr cudnnDataType_t data_type = cudnnTypeWrapper<T>::type;
          SetTensorDescriptor(data_type, order_, X_dims, &X_desc_);
          SetTensorDescriptor(data_type, order_, Y_dims, &Y_desc_);
          cached_X_dims_ = X_dims;
        }
        CUDNN_ENFORCE(cudnnPoolingForward(
            cudnn_wrapper_.inline_cudnn_handle(),
            pooling_desc_,
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            X_data,
            cudnnTypeWrapper<T>::kZero(),
            Y_desc_,
            Y_data));

        return true;
        */
    }
}

impl<Functor> Drop for CudnnPoolOp<Functor> {
    fn drop(&mut self) {
        todo!();
        /* 
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
        CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
       */
    }
}

///-----------------------------------

pub struct CudnnPoolGradientOp<Functor> {
    base: ConvPoolOpBase<CUDAContext>,

    cudnn_wrapper:   CudnnWrapper,
    x_desc:          CudnnTensorDescriptor,
    y_desc:          CudnnTensorDescriptor,
    pooling_desc:    CudnnPoolingDescriptor,
    functor:         Functor,
    equal_padding:   bool,
    cached_X_dims:   Vec<i64>,
}

impl<Functor> Drop for CudnnPoolGradientOp<Functor> {
    fn drop(&mut self) {
        todo!();
        /* 
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
        CUDNN_ENFORCE(cudnnDestroyPoolingDescriptor(pooling_desc_));
       */
    }
}

impl<Functor> CudnnPoolGradientOp<Functor> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            functor_(*this),
            equal_padding_(std::equal(
                pads_.cbegin(),
                pads_.cbegin() + kernel_.size(),
                pads_.cbegin() + kernel_.size())) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
        CUDNN_ENFORCE(cudnnCreatePoolingDescriptor(&pooling_desc_));
        if (!global_pooling_ && equal_padding_) {
          if (kernel_.size() == 2) {
            CUDNN_ENFORCE(cudnnSetPooling2dDescriptor(
                pooling_desc_,
                functor_.GetPoolingMode(),
                CUDNN_NOT_PROPAGATE_NAN,
                kernel_h(),
                kernel_w(),
                pad_t(),
                pad_l(),
                stride_h(),
                stride_w()));
          } else if (kernel_.size() == 3) {
            CUDNN_ENFORCE(cudnnSetPoolingNdDescriptor(
                pooling_desc_,
                functor_.GetPoolingMode(),
                CUDNN_NOT_PROPAGATE_NAN,
                kernel_.size(),
                kernel_.data(),
                pads_.data(),
                stride_.data()));
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
        const auto& Y = Input(1);
        const auto& dY = Input(2);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const std::vector<int> X_HW_dims = GetDims(X);
        const std::vector<int> Y_HW_dims = GetDims(Y);
        ConvPoolOpBase<CUDAContext>::ComputePads(X_HW_dims);
        const T* dY_data = dY.template data<T>();
        const T* X_data = X.template data<T>();
        const T* Y_data = Y.template data<T>();
        T* dX_data = dX->template mutable_data<T>();

        if (N == 0) {
          return true;
        }

        if (global_pooling_) {
          const int HxW = X.numel() / (N * C);
          if (order_ == StorageOrder::NCHW) {
            return functor_.template GlobalPoolingBackward<T, StorageOrder::NCHW>(
                N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
          } else {
            return functor_.template GlobalPoolingBackward<T, StorageOrder::NHWC>(
                N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
          }
        }

        if (order_ == StorageOrder::NHWC) {
          // Cudnn Pooling on NHWC order is very slow, fallback to CUDA
          // implementation.
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
        } else if (!equal_padding_ || ndim == 3) {
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
        }

        const std::vector<std::int64_t> X_dims = X.sizes().vec();
        const std::vector<std::int64_t> Y_dims = Y.sizes().vec();
        if (cached_X_dims_ != X_dims) {
          constexpr cudnnDataType_t data_type = cudnnTypeWrapper<T>::type;
          SetTensorDescriptor(data_type, order_, X_dims, &X_desc_);
          SetTensorDescriptor(data_type, order_, Y_dims, &Y_desc_);
          cached_X_dims_ = X_dims;
        }
        CUDNN_ENFORCE(cudnnPoolingBackward(
            cudnn_wrapper_.inline_cudnn_handle(),
            pooling_desc_,
            cudnnTypeWrapper<T>::kOne(),
            Y_desc_,
            Y_data,
            Y_desc_,
            dY_data,
            X_desc_,
            X_data,
            cudnnTypeWrapper<T>::kZero(),
            X_desc_,
            dX_data));

        return true;
        */
    }
}

///----------------------------
pub struct CudnnAveragePoolFunctor {
    avg_pool_functor: AveragePoolFunctor<CUDAContext>,
}

impl CudnnAveragePoolFunctor {
    
    pub fn new(op: &OperatorStorage) -> Self {
        todo!();
        /*
            : avg_pool_functor(op)
        */
    }
    
    #[inline] pub fn get_pooling_mode(&self) -> CudnnPoolingMode {
        
        todo!();
        /*
            return avg_pool_functor.count_include_pad
            ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
            : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        x:         *const T,
        y:         *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.GlobalPoolingForward<T, kOrder>(
              N, C, HxW, X, Y, context);
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:          i32,
        c:          i32,
        x_dims:     &Vec<i32>,
        y_dims:     &Vec<i32>,
        kernel:     &Vec<i32>,
        dilation:   &Vec<i32>,
        stride:     &Vec<i32>,
        pads:       &Vec<i32>,
        x:          *const T,
        y:          *mut T,
        context:    *mut CUDAContext) -> bool {
        todo!();
        /*
            return avg_pool_functor.Forward<T, kOrder>(
              N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        dY:        *const T,
        x:         *const T,
        y:         *const T,
        dX:        *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.GlobalPoolingBackward<T, kOrder>(
              N, C, HxW, dY, X, Y, dX, context);
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:           i32,
        c:           i32,
        x_dims:      &Vec<i32>,
        y_dims:      &Vec<i32>,
        kernel:      &Vec<i32>,
        dilation:    &Vec<i32>,
        stride:      &Vec<i32>,
        pads:        &Vec<i32>,
        dY:          *const T,
        x:           *const T,
        y:           *const T,
        dX:          *mut T,
        context:     *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.Backward<T, kOrder>(
              N,
              C,
              X_dims,
              Y_dims,
              kernel,
              dilation,
              stride,
              pads,
              dY,
              X,
              Y,
              dX,
              context);
        */
    }
}

///----------------------------
pub struct CudnnMaxPoolFunctor {
    max_pool_functor: MaxPoolFunctor<CUDAContext>,
    deterministic: bool,
}

impl CudnnMaxPoolFunctor {
    
    pub fn new(op: &OperatorStorage) -> Self {
        todo!();
        /*
            : max_pool_functor(op),
            deterministic(op.GetSingleArgument<bool>("deterministic", false))
        */
    }
    
    #[inline] pub fn get_pooling_mode(&self) -> CudnnPoolingMode {
        
        todo!();
        /*
            #if CUDNN_VERSION_MIN(6, 0, 0)
        return deterministic ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_MAX;
    #else
        return CUDNN_POOLING_MAX;
    #endif
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &mut self, n: i32,
        c: i32,
        hxW: i32,
        x: *const T,
        y: *mut T,
        context: *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.GlobalPoolingForward<T, kOrder>(
              N, C, HxW, X, Y, context);
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        x_dims:    &Vec<i32>,
        y_dims:    &Vec<i32>,
        kernel:    &Vec<i32>,
        dilation:  &Vec<i32>,
        stride:    &Vec<i32>,
        pads:      &Vec<i32>,
        x:         *const T,
        y:         *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.Forward<T, kOrder>(
              N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:        i32,
        c:        i32,
        hxW:      i32,
        dY:       *const T,
        x:        *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.GlobalPoolingBackward<T, kOrder>(
              N, C, HxW, dY, X, Y, dX, context);
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:        i32,
        c:        i32,
        x_dims:   &Vec<i32>,
        y_dims:   &Vec<i32>,
        kernel:   &Vec<i32>,
        dilation: &Vec<i32>,
        stride:   &Vec<i32>,
        pads:     &Vec<i32>,
        dY:       *const T,
        x:        *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return max_pool_functor.Backward<T, kOrder>(
              N,
              C,
              X_dims,
              Y_dims,
              kernel,
              dilation,
              stride,
              pads,
              dY,
              X,
              Y,
              dX,
              context);
        */
    }
}

register_cudnn_operator!{AveragePool,              CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePoolGradient,      CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool1D,            CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool1DGradient,    CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool2D,            CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool2DGradient,    CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool3D,            CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool3DGradient,    CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{MaxPool,                  CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPoolGradient,          CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool1D,                CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool1DGradient,        CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool2D,                CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool2DGradient,        CudnnPoolGradientOp<CudnnMaxPoolFunctor>}

register_cudnn_operator!{MaxPool3D,                CudnnPoolOp<CudnnMaxPoolFunctor>}
register_cudnn_operator!{MaxPool3DGradient,        CudnnPoolGradientOp<CudnnMaxPoolFunctor>}
