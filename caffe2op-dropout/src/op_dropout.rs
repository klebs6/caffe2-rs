crate::ix!();

///----------------------------------------
#[test] fn dropout_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Dropout",
        ["X"],
        ["Y"] + ["mask"],
        ratio=0.5,
        is_test=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(5, 5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("mask:", workspace.FetchBlob("mask"))

    **Result**

    X: [[5. 4. 3. 6. 9.]
     [2. 1. 8. 0. 9.]
     [7. 3. 0. 6. 3.]
     [1. 8. 2. 6. 4.]
     [6. 2. 6. 4. 0.]]
    Y: [[ 0.  0.  0. 12. 18.]
     [ 0.  0. 16.  0.  0.]
     [ 0.  0.  0. 12.  6.]
     [ 0.  0.  4.  0.  0.]
     [12.  0.  0.  0.  0.]]
    mask: [[False False False  True  True]
     [False False  True  True False]
     [False False  True  True  True]
     [False False  True False False]
     [ True False False False False]]
    */
}

/**
  | `Dropout` takes one input data tensor
  | (`X`) and produces two tensor outputs,
  | `Y` and `mask`.
  | 
  | If the `is_test` argument is zero (default=0),
  | the output `Y` will be the input with
  | random elements zeroed.
  | 
  | The probability that a given element
  | is zeroed is determined by the `ratio`
  | argument.
  | 
  | If the `is_test` argument is set to non-zero,
  | the output `Y` is exactly the same as
  | the input `X`.
  | 
  | -----------
  | @note
  | 
  | outputs are scaled by a factor of $\frac{1}{1-ratio}$
  | during training, so that during test
  | time, we can simply compute an identity
  | function. This scaling is important
  | because we want the output at test time
  | to equal the expected value at training
  | time.
  | 
  | Dropout has been proven to be an effective
  | regularization technique to prevent
  | overfitting during training.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc
  |
  */
pub struct DropoutOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    ratio:   f32,
    is_test: bool,

    /**
      | Input: X;
      | 
      | Output: Y, mask.
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{Dropout, 1}

num_outputs!{Dropout, (1,2)}

inputs!{Dropout, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Dropout, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor."),
    1 => ("mask", "*(type: Tensor`<bool>`)* The output mask containing boolean values for each element, signifying which elements are dropped out. If `is_test` is nonzero, this output is not filled.")
}

args!{Dropout, 
    0 => ("ratio", "*(type: float; default: 0.5)* Probability of an element to be zeroed.")
}

inherit_onnx_schema!{Dropout}

tensor_inference_function!{Dropout, /*[](const OperatorDef& def,
    const vector<TensorShape>& in) {
    CAFFE_ENFORCE_EQ(1, in.size());
    vector<TensorShape> out;
    ArgumentHelper argsHelper(def);
    out.push_back(in[0]);
    if (def.output().size() == 2) {
        out.push_back(in[0]);
        out[1].set_data_type(TensorProto_DataType_BOOL);
    }
    return out;
    }*/
}

allow_inplace!{Dropout, vec![(0, 0)]}

arg_is_test!{Dropout, 
    "*(type: int; default: 0)* If zero (train mode), perform dropout. If non-zero (test mode), Y = X."
}

impl<T, Context> DropoutOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
            is_test_(this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        */
    }
}

impl DropoutOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      if (is_test_) {
        if (!IsInputOutputAlias(0, 0)) {
          context_.CopyFromCPU<float>(
              X.numel(), X.data<float>(), Y->template mutable_data<float>());
        }
        return true;
      } else {
        float scale = 1. / (1. - ratio_);
        // mask=true means keep, and mask=false means not keep, so we will
        // generate probability depending on 1-ratio.
        at::bernoulli_distribution<double> dist(1. - ratio_);
        const float* Xdata = X.data<float>();
        float* Ydata = Y->template mutable_data<float>();

        auto mask = Output(1, X.sizes(), at::dtype<bool>());
        bool* mask_data = mask->template mutable_data<bool>();
        auto* gen = context_.RandGenerator();
        for (int i = 0; i < X.numel(); ++i) {
          mask_data[i] = dist(gen) > 0.5;
          Ydata[i] = Xdata[i] * scale * mask_data[i];
        }
        return true;
      }
        */
    }
}


///----------------------------------------
pub struct DropoutGradientOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    ratio:   f32,
    is_test: bool,

    /**
      | Input: dY, mask;
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

impl<T,Context> DropoutGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
            is_test_(this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        */
    }
}

impl DropoutGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      if (is_test_) {
        if (dX != &dY) {
          context_.CopyFromCPU<float>(
              dY.numel(), dY.data<float>(), dX->template mutable_data<float>());
        }
        return true;
      } else {
        auto& mask = Input(1);
        CAFFE_ENFORCE_EQ(dY.numel(), mask.numel());
        const float* dYdata = dY.data<float>();
        const bool* mask_data = mask.data<bool>();
        float* dXdata = dX->template mutable_data<float>();
        float scale = 1. / (1. - ratio_);
        for (int i = 0; i < dY.numel(); ++i) {
          dXdata[i] = dYdata[i] * mask_data[i] * scale;
        }
        return true;
      }
        */
    }
}

register_cpu_operator!{Dropout, DropoutOp<float, CPUContext>}

register_cpu_gradient_operator!{
    DropoutGrad,
    DropoutGradientOp<f32, CPUContext>
}

num_inputs!{DropoutGrad, (1,2)}

num_outputs!{DropoutGrad, 1}

allow_inplace!{DropoutGrad, vec![(0, 0)]}

pub struct GetDropoutGradient;

impl GetGradientDefs for GetDropoutGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argshelper(def_);
        auto is_test = argshelper.GetSingleArgument<bool>("is_test", 0);
        if (is_test) {
          return SingleGradientDef(
              "DropoutGrad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return SingleGradientDef(
              "DropoutGrad",
              "",
              vector<string>{GO(0), O(1)},
              vector<string>{GI(0)});
        }
        */
    }
}

register_gradient!{Dropout, GetDropoutGradient}

/**
  | cudnnRestoreDropoutDescriptor is
  | needed for correctness and doesn't
  | exist prior to cuDNN v7
  |
  */
#[cfg(cudnn_version_min = "7.0.0")]
pub struct CudnnDropoutOp {

    //USE_OPERATOR_FUNCTIONS(CUDAContext);
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

///------------------------------------------------------

pub struct CudnnDropoutGradientOp {

    //USE_OPERATOR_FUNCTIONS(CUDAContext);
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
