crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnTransposeOp {
    storage:       OperatorStorage,
    context:       CUDAContext,
    cudnn_wrapper: CudnnWrapper,
    x_desc:        CudnnTensorDescriptor,
    y_desc:        CudnnTensorDescriptor,
    cached_dtype:  CudnnDataType, // = cudnnTypeWrapper<float>::type;
    cached_X_dims: Vec<i64>,
    axes:          Vec<i32>,
}

impl Drop for CudnnTransposeOp {

    fn drop(&mut self) {
        //CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
        //CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
    }
}

impl CudnnTransposeOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            axes_(OperatorStorage::GetRepeatedArgument<int>("axes")) 

        // Checks the legality of axes_: it should be from 0 to axes_.size().
        std::vector<int> axes_sorted(axes_);
        std::sort(axes_sorted.begin(), axes_sorted.end());
        for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
          if (axes_sorted[i] != i) {
            CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
          }
        }

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, int>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const int ndim = X.dim();
            if (axes_.empty()) {
              axes_.resize(ndim);
              std::iota(axes_.rbegin(), axes_.rend(), 0);
            } else {
              CAFFE_ENFORCE_EQ(axes_.size(), ndim);
            }
            std::vector<std::int64_t> X_dims = X.sizes().vec();
            std::vector<std::int64_t> Y_dims(ndim);
            for (int i = 0; i < ndim; ++i) {
              Y_dims[i] = X_dims[axes_[i]];
            }
            auto* Y = Output(0, Y_dims, at::dtype<T>());
            const T* X_data = X.template data<T>();
            T* Y_data = Y->template mutable_data<T>();
            if (X.numel() == 0) {
              return true;
            }
            if (!IsFloatType<T>() || !IsCudnnValidTensor(X)) {
              math::Transpose<std::int64_t, T, CUDAContext>(
                  ndim, X_dims.data(), axes_.data(), X_data, Y_data, &context_);
              return true;
            }
            if (cudnnTypeWrapper<T>::type != cached_dtype_ ||
                X_dims != cached_X_dims_) {
              SetTensorDescriptor(cudnnTypeWrapper<T>::type, X_dims, Y_dims);
              cached_dtype_ = cudnnTypeWrapper<T>::type;
              cached_X_dims_ = X_dims;
            }
            CUDNN_ENFORCE(cudnnTransformTensor(
                cudnn_wrapper_.inline_cudnn_handle(),
                cudnnTypeWrapper<T>::kOne(),
                X_desc_,
                X_data,
                cudnnTypeWrapper<T>::kZero(),
                Y_desc_,
                Y_data));
            return true;
        */
    }
    #[inline] pub const fn is_float_type<T>() -> bool {
        todo!();
        /*
            return std::is_same<T, float>::value || std::is_same<T, double>::value ||
                std::is_same<T, at::Half>::value;
        */
    }
    
    #[inline] pub fn is_cu_dnnvalid_tensor(&self, x: &Tensor) -> bool {
        
        todo!();
        /*
            const int ndim = X.dim();
        return ndim >= 3 && ndim <= CUDNN_DIM_MAX &&
            X.numel() < int32_t::max;
        */
    }
    
    #[inline] pub fn set_tensor_descriptor(
        &mut self, 
        data_type: CudnnDataType,
        x_dims: &Vec<i64>,
        y_dims: &Vec<i64>)  
    {
        todo!();
        /*
            const int ndim = X_dims.size();
        std::vector<int> dims(Y_dims.cbegin(), Y_dims.cend());
        std::vector<int> X_strides(ndim);
        std::vector<int> X_buff(ndim);
        std::vector<int> Y_strides(ndim);
        X_buff.back() = 1;
        Y_strides.back() = 1;
        for (int i = ndim - 1; i > 0; --i) {
          X_buff[i - 1] = X_buff[i] * X_dims[i];
          Y_strides[i - 1] = Y_strides[i] * Y_dims[i];
        }
        for (int i = 0; i < ndim; ++i) {
          X_strides[i] = X_buff[axes_[i]];
        }
        CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
            X_desc_, data_type, ndim, dims.data(), X_strides.data()));
        CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
            Y_desc_, data_type, ndim, dims.data(), Y_strides.data()));
        */
    }
    
    // Cudnn 5.1 does not have int support yet.
    #[cfg(not(cudnn_version_min_gt_6_0_0))]
    #[inline] pub fn do_run_with_type_int(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const int ndim = X.dim();
      if (axes_.empty()) {
        axes_.resize(ndim);
        std::iota(axes_.rbegin(), axes_.rend(), 0);
      } else {
        CAFFE_ENFORCE_EQ(axes_.size(), ndim);
      }
      std::vector<std::int64_t> X_dims = X.sizes().vec();
      std::vector<std::int64_t> Y_dims(ndim);
      for (int i = 0; i < ndim; ++i) {
        Y_dims[i] = X_dims[axes_[i]];
      }
      auto* Y = Output(0, Y_dims, at::dtype<T>());
      const T* X_data = X.template data<T>();
      T* Y_data = Y->template mutable_data<T>();
      math::Transpose<std::int64_t, T, CUDAContext>(
          ndim, X_dims.data(), axes_.data(), X_data, Y_data, &context_);
      return true;
        */
    }
}

register_cudnn_operator!{Transpose, CudnnTransposeOp}
