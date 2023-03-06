crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnLRNOp {

    storage: OperatorStorage,
    context: CUDAContext,

    cudnn_wrapper:     CudnnWrapper,
    data_desc:         CudnnTensorDescriptor,
    norm_desc:         CudnnLRNDescriptor,
    cudnn_input_dims:  Vec<i64>,
    size:              i32,
    alpha:             f32,
    beta:              f32,
    bias:              f32,

    /*
      | Input: X,
      | 
      | Output: Y
      |
      */
}

impl Drop for CudnnLRNOp {

    fn drop(&mut self) {
        todo!();
        /*
           CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
           CUDNN_ENFORCE(cudnnDestroyLRNDescriptor(norm_desc_));
        */
    }
}

impl CudnnLRNOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            size_(OperatorStorage::GetSingleArgument<int>("size", 0)),
            alpha_(OperatorStorage::GetSingleArgument<float>("alpha", 0)),
            beta_(OperatorStorage::GetSingleArgument<float>("beta", 0)),
            bias_(OperatorStorage::GetSingleArgument<float>("bias", 1)) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

        CUDNN_ENFORCE(cudnnCreateLRNDescriptor(&norm_desc_));
        CUDNN_ENFORCE(
            cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, bias_));
        */
    }

    #[inline] pub fn do_run_with_type<T, M>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
          auto* Y = Output(0);

          // Reshape tensor descriptors if necessary
          if (X.sizes() != cudnn_input_dims_) {
            VLOG(1) << "Setting descriptors";
            cudnn_input_dims_ = X.sizes().vec();
            int C = 1, H = 1, W = 1;
            // Normal 4-dimensional tensors for images.
            C = X.dim32(1);
            H = X.dim32(2);
            W = X.dim32(3);
            CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                data_desc_,
                GetCudnnTensorFormat(StorageOrder::NCHW),
                cudnnTypeWrapper<T>::type,
                X.dim32(0),
                C,
                H,
                W));
          }

          // now actually run the computation
          CUDNN_ENFORCE(cudnnLRNCrossChannelForward(
              cudnn_wrapper_.inline_cudnn_handle(),
              norm_desc_,
              CUDNN_LRN_CROSS_CHANNEL_DIM1,
              cudnnTypeWrapper<T>::kOne(),
              data_desc_,
              X.template data<T>(),
              cudnnTypeWrapper<T>::kZero(),
              data_desc_,
              Y->template mutable_data<T>()));

          return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // dispatch based on contents of tensor(s)
      const auto& X = Input(0);
      auto* Y = Output(0);
      Y->ResizeLike(X);

      if (X.IsType<float>()) {
        return DoRunWithType<float, float>();
      } else if (X.IsType<at::Half>()) {
        return DoRunWithType<at::Half, float>();
      } else {
        CAFFE_THROW("Unsupported input type");
      }
      return false;
        */
    }
}
