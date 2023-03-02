crate::ix!();

#[USE_OPERATOR_FUNCTIONS(CUDAContext)]
pub struct CudnnDropoutGradientOp {

    storage:                        OperatorStorage,
    context:                        CUDAContext,

    cudnn_wrapper:                  CudnnWrapper,
    data_desc:                      CudnnTensorDescriptor,
    dropout_desc:                   CudnnDropoutDescriptor,
    cudnn_input_dims:               Vec<i64>,
    scratch_blob:                   *mut Blob,
    ratio:                          f32,
    is_test:                        bool,
    states_size_in_bytes:           usize,
    reserve_space_size_in_bytes:    usize,

    /**
      | only need to initialize states once
      | (size is static)
      |
      */
    states_initialized:             bool,
    random_seed:                    u64,

    /*
      | Input: dY, mask_and_states,
      | 
      | Output: dX
      |
      */
}

impl CudnnDropoutGradientOp {
    
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

        // Share scratch with the forward op
        scratch_blob_ =
            ws->GetBlob(CudnnDropoutOp::scratch_blob_name(operator_def.input(1)));
        CAFFE_ENFORCE(scratch_blob_);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // dispatch based on contents of tensor(s)
      const auto& dY = Input(0);
      auto* dX = Output(0);

      dX->ResizeLike(dY);

      if (dY.IsType<float>()) {
        return DoRunWithType<float, float>();
      } else if (dY.IsType<at::Half>()) {
        return DoRunWithType<at::Half, float>();
      }
      return false;
        */
    }

    #[inline] pub fn do_run_with_type<T, M>(&mut self) -> bool {
        todo!();
        /*
            const auto& dY = Input(0);
          const auto& mask = Input(1);
          const Tensor& states = scratch_blob_->Get<Tensor>();
          auto* dX = Output(0);

          auto size_prod = 1;
          for (auto dim : dY.sizes()) {
            size_prod *= dim;
          }

          if (!states_initialized_) {
            // set the dropout descriptor
            {
              // Need to protect  as clashes with NCCL
              std::lock_guard<std::mutex> lk(CUDAContext::mutex());
              CUDNN_ENFORCE(cudnnRestoreDropoutDescriptor(
                  dropout_desc_,
                  cudnn_wrapper_.inline_cudnn_handle(),
                  ratio_,
                  const_cast<uint8_t*>(states.data<uint8_t>()),
                  states_size_in_bytes_,
                  random_seed_
                  ));
            }
            states_initialized_ = true;
          }

          if (dY.sizes() != cudnn_input_dims_) {
            cudnn_input_dims_ = dY.sizes().vec();
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
          }

          // run the computation
          void* mask_data = const_cast<void*>(mask.raw_data());
          CUDNN_ENFORCE(cudnnDropoutBackward(
              cudnn_wrapper_.inline_cudnn_handle(),
              dropout_desc_,
              data_desc_,
              dY.data<T>(),
              data_desc_,
              dX->template mutable_data<T>(),
              mask_data,
              reserve_space_size_in_bytes_));
          return true;
        */
    }
}

impl Drop for CudnnDropoutGradientOp {
    fn drop(&mut self) {
        //CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
        //CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
}

register_cudnn_operator!{Dropout, CudnnDropoutOp}

register_cudnn_operator!{DropoutGrad, CudnnDropoutGradientOp}
