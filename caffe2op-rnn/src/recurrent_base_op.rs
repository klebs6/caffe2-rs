crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct RecurrentBaseOp<T> {
    storage:            OperatorStorage,
    context:            CUDAContext,
    cudnn_wrapper:      CudnnWrapper,
    dropout_desc:       CudnnDropoutDescriptor,
    rnn_desc:           CudnnRNNDescriptor,
    w_desc:             CudnnFilterDescriptor,
    hx_desc:            CudnnTensorDescriptor,
    cx_desc:            CudnnTensorDescriptor,
    hy_desc:            CudnnTensorDescriptor,
    cy_desc:            CudnnTensorDescriptor,
    x_desc:             Box<TensorDescriptors<T>>,
    y_desc:             Box<TensorDescriptors<T>>,
    cached_input_dims:  Vec<i64>,
    reserve_nbytes:     usize,
    cudnn_ws_nbytes:    usize,
    phantom:            PhantomData<T>,
}

impl<T> RecurrentBaseOp<T> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...), cudnn_wrapper_(&context_) 

          CUDNN_ENFORCE(cudnnCreateDropoutDescriptor(&dropoutDesc_));
          CUDNN_ENFORCE(cudnnCreateRNNDescriptor(&rnnDesc_));
          CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&wDesc_));
          CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&hxDesc_));
        */
    }
}

#[macro_export] macro_rules! use_recurrent_base_functions {
    () => {
        todo!();
        /*
        
          USE_OPERATOR_FUNCTIONS(CUDAContext);        
          using RecurrentBaseOp<T>::cudnn_wrapper_;   
          using RecurrentBaseOp<T>::dropoutDesc_;     
          using RecurrentBaseOp<T>::rnnDesc_;         
          using RecurrentBaseOp<T>::wDesc_;           
          using RecurrentBaseOp<T>::hxDesc_;          
          using RecurrentBaseOp<T>::cxDesc_;          
          using RecurrentBaseOp<T>::hyDesc_;          
          using RecurrentBaseOp<T>::cyDesc_;          
          using RecurrentBaseOp<T>::xDesc_;           
          using RecurrentBaseOp<T>::yDesc_;           
          using RecurrentBaseOp<T>::cachedInputDims_; 
          using RecurrentBaseOp<T>::reserveNbytes_;   
          using RecurrentBaseOp<T>::cudnnWsNbytes_;   
          using RecurrentBaseOp<T>::initialize;
        */
    }
}

impl<T> Drop for RecurrentBaseOp<T> {

    fn drop(&mut self) {
        todo!();
        /* 
          CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropoutDesc_));
          CUDNN_ENFORCE(cudnnDestroyRNNDescriptor(rnnDesc_));
          CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(wDesc_));
          CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(hxDesc_));
        */
    }
}

impl<T> RecurrentBaseOp<T> {
    
    #[inline] pub fn initialize(&mut self, 
        input:          &Tensor,
        dropout_states: *mut Tensor,
        // If passed, reshapes to the appropriate size
        output:         *mut Tensor,
        hidden_output:  *mut Tensor,
        cell_output:    *mut Tensor)  {

        todo!();
        /*
            static_assert(sizeof(T) == 4, ""); // workaround clang bug
      CAFFE_ENFORCE_GE(input.dim(), 3);
      const int seqLength = input.size(0);
      const int batchSize = input.size(1);
      const int inputDim = input.size(2);
      const int hiddenSize = OperatorStorage::GetSingleArgument<int>("hidden_size", 0);
      CAFFE_ENFORCE_GT(hiddenSize, 0);
      const auto bidirectional =
          OperatorStorage::GetSingleArgument<int>("bidirectional", 0);
      CAFFE_ENFORCE(bidirectional == 0 || bidirectional == 1);
      const auto numDirections = bidirectional == 1 ? 2 : 1;
      const auto outputDim = hiddenSize * numDirections;
      const auto rnnDirection =
          bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
      const auto numLayers = OperatorStorage::GetSingleArgument<int>("num_layers", 0);
      CAFFE_ENFORCE_GT(numLayers, 0);
      const auto& rnnModeStr =
          OperatorStorage::GetSingleArgument<string>("rnn_mode", "");
      CAFFE_ENFORCE(rnnModeStr == "lstm" || rnnModeStr == "gru");
      const auto rnnMode = rnnModeStr == "lstm" ? CUDNN_LSTM : CUDNN_GRU;
      const auto& rnnInputStr =
          OperatorStorage::GetSingleArgument<string>("input_mode", "");
      CAFFE_ENFORCE(rnnInputStr == "linear" || rnnInputStr == "skip");
      const auto rnnInput =
          rnnInputStr == "linear" ? CUDNN_LINEAR_INPUT : CUDNN_SKIP_INPUT;

      // Dropout setup
      {
        if (dropoutStates) {
          size_t stateSize;
          float dropout_param =
              OperatorStorage::GetSingleArgument<float>("dropout", 1.0);
          if (dropout_param < 1.0) {
            CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
                cudnn_wrapper_.inline_cudnn_handle(), &stateSize));
            dropoutStates->Resize(std::vector<int>{static_cast<int>(
                stateSize / 4 /* sizeof(T) - workaround clang bug */)});
            CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
                dropoutDesc_,
                cudnn_wrapper_.inline_cudnn_handle(),
                dropout_param,
                dropoutStates->template mutable_data<T>(),
                stateSize,
                OperatorStorage::GetSingleArgument<int>("seed", 0)));
          }
        }
      }

      // RNN setup
      {
    #if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_ENFORCE(cudnnSetRNNDescriptor_v6(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            hiddenSize,
            numLayers,
            dropoutDesc_,
            rnnInput,
            rnnDirection,
            rnnMode,
            CUDNN_RNN_ALGO_STANDARD, // TODO: verify correctness / efficiency.
            cudnnTypeWrapper<T>::type));
    #else
        CUDNN_ENFORCE(cudnnSetRNNDescriptor(
            rnnDesc_,
            hiddenSize,
            numLayers,
            dropoutDesc_,
            rnnInput,
            rnnDirection,
            rnnMode,
            cudnnTypeWrapper<T>::type));
    #endif
      }
      // X setup
      {
        xDesc_.reset(new detail::TensorDescriptors<T>(
            seqLength,
            // Third dimension is unused
            {batchSize, inputDim, 1},
            // Fully-packed
            {inputDim, 1, 1}));
      }
      // Y setup
      {
        yDesc_.reset(new detail::TensorDescriptors<T>(
            seqLength,
            // Third dimension is unused
            {batchSize, hiddenSize * numDirections, 1},
            // Fully-packed
            {numDirections * hiddenSize, 1, 1}));

        if (output) {
          output->Resize(std::vector<int>{seqLength, batchSize, outputDim});
        }
      }

      // Hidden/Cell setup
      {
        const std::array<int, 3> dim{
            numLayers * numDirections, batchSize, hiddenSize};
        const std::array<int, 3> stride{batchSize * hiddenSize, hiddenSize, 1};
        CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
            hxDesc_, cudnnTypeWrapper<T>::type, 3, dim.data(), stride.data()));
        cxDesc_ = hxDesc_;
        hyDesc_ = hxDesc_;
        cyDesc_ = hxDesc_;

        if (hiddenOutput) {
          hiddenOutput->Resize(
              std::vector<int>{numLayers * numDirections, batchSize, hiddenSize});
        }

        if (cellOutput) {
          cellOutput->Resize(
              std::vector<int>{numLayers * numDirections, batchSize, hiddenSize});
        }
      }

      // Weights setup
      {
        size_t weightsSize;
        CUDNN_ENFORCE(cudnnGetRNNParamsSize(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            xDesc_->descs()[0],
            &weightsSize,
            cudnnTypeWrapper<T>::type));
        const std::array<int, 3> dims{
            static_cast<int>(
                weightsSize / 4 /* sizeof(T) - workaround clang bug */),
            1,
            1};
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            wDesc_, cudnnTypeWrapper<T>::type, CUDNN_TENSOR_NCHW, 3, dims.data()));
      }

      // RNN workspace size
      {
        CUDNN_ENFORCE(cudnnGetRNNWorkspaceSize(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            seqLength,
            xDesc_->descs(),
            &cudnnWsNbytes_));
      }
        */
    }
}
