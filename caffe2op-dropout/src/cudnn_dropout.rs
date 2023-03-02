crate::ix!();

/**
  | cudnnRestoreDropoutDescriptor is
  | needed for correctness and doesn't
  | exist prior to cuDNN v7
  |
  */
#[cfg(cudnn_version_min = "7.0.0")]
#[USE_OPERATOR_FUNCTIONS(CUDAContext)]
pub struct CudnnDropoutOp {

    context:                     Operator<Context>,

    cudnn_wrapper:               CudnnWrapper,
    data_desc:                   CudnnTensorDescriptor,
    dropout_desc:                CudnnDropoutDescriptor,
    cudnn_input_dims:            Vec<i64>,
    ratio:                       f32,
    is_test:                     bool,
    scratch_blob:                *mut Blob,
    states_size_in_bytes:        usize,
    reserve_space_size_in_bytes: usize,

    /**
      | track whether states have been initialized
      | only needs to happen once
      |
      */
    states_initialized:          bool,

    /// random seed
    random_seed:                 u64,

    /*
      | Input: X,
      | 
      | Output: Y, mask_and_states
      |
      */
}

#[cfg(cudnn_version_min = "7.0.0")]
impl CudnnDropoutOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CUDAContext>(operator_def, ws),
            cudnn_wrapper_(&context_),
            ratio_(OperatorStorage::GetSingleArgument<float>("ratio", 0.5)),
            is_test_(OperatorStorage::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
            states_initialized_(false),
            random_seed_(operator_def.device_option().random_seed()) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));

        CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropout_desc_));
        CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
            cudnn_wrapper_.inline_cudnn_handle(),
            reinterpret_cast<size_t*>(&states_size_in_bytes_)));

        if (!is_test_) {
          scratch_blob_ = ws->CreateBlob(scratch_blob_name(operator_def.output(1)));
          CAFFE_ENFORCE(scratch_blob_);
        }
        */
    }
    
    #[inline] pub fn scratch_blob_name(mask_blob_name: String) -> String {
        
        todo!();
        /*
            return "cudnn_dropout_scratch_" + mask_blob_name;
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
      }
      return false;
        */
    }

    #[inline] pub fn do_run_with_type<T, M>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
          auto* Y = Output(0);

          auto size_prod = 1;
          for (auto dim : X.sizes()) {
            size_prod *= dim;
          }
          // now actually run the computation
          if (is_test_) {
            if (Y != &X) {
              context_.CopySameDevice<T>(
                  X.numel(), X.template data<T>(), Y->template mutable_data<T>());
            }
            return true;
          } else {
            // Reshape tensor descriptors if necessary
            if (X.sizes() != cudnn_input_dims_) {
              CAFFE_ENFORCE(scratch_blob_);
              Tensor* states = BlobGetMutableTensor(scratch_blob_, CUDA);
              cudnn_input_dims_ = X.sizes().vec();
              CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                  data_desc_,
                  GetCudnnTensorFormat(StorageOrder::NCHW),
                  cudnnTypeWrapper<T>::type,
                  size_prod,
                  1,
                  1,
                  1));

              // get the reserve space we need
              CUDNN_ENFORCE(cudnnDropoutGetReserveSpaceSize(
                  data_desc_, &reserve_space_size_in_bytes_));

              states->Resize(states_size_in_bytes_);

              if (!states_initialized_) {
                // set the dropout descriptor (note: need to allocate the states data
                // before acquiring the mutex)
                uint8_t* states_data = states->template mutable_data<uint8_t>();
                {
                  // Need to protect  as clashes with NCCL
                  std::lock_guard<std::mutex> lk(CUDAContext::mutex());
                  CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
                      dropout_desc_,
                      cudnn_wrapper_.inline_cudnn_handle(),
                      ratio_,
                      states_data,
                      states_size_in_bytes_,
                      random_seed_
                      ));
                }
                states_initialized_ = true;
              }
            }
            auto* mask = Output(
                1,
                {static_cast<int64_t>(reserve_space_size_in_bytes_)},
                at::dtype<uint8_t>());
            CUDNN_ENFORCE(cudnnDropoutForward(
                cudnn_wrapper_.inline_cudnn_handle(),
                dropout_desc_,
                data_desc_,
                X.template data<T>(),
                data_desc_,
                Y->template mutable_data<T>(),
                mask->template mutable_data<uint8_t>(),
                reserve_space_size_in_bytes_));
          }
          return true;
        */
    }
}

#[cfg(cudnn_version_min = "7.0.0")]
impl Drop for CudnnDropoutOp {

    fn drop(&mut self) {
        //CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
        //CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
}
