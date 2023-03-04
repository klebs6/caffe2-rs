crate::ix!();

/**
  | Computes the result of passing an input
  | vector
  | 
  | X into a fully connected layer with 2D
  | weight matrix W and 1D bias vector b.
  | 
  | That is, the layer computes Y = X * W^T
  | + b, where
  | 
  | X has size (M x K), W has size (N x K), b has
  | size (N), and Y has size (M x N), where
  | M is often the batch size.
  | 
  | -----------
  | @note
  | 
  | X does not need to explicitly be a 2D vector;
  | rather, it will be coerced into one.
  | 
  | For an arbitrary n-dimensional tensor
  | X \in [a_0, a_1 * ... * a_{n-1}].
  | 
  | Only this case is supported! Lastly,
  | even though b is a 1D vector of size N,
  | it is copied/resized to be size (M x N)
  | implicitly and added to each vector
  | in the batch.
  | 
  | Each of these dimensions must be matched
  | correctly, or else the operator will
  | throw errors.
  |
  */
pub struct Int8FCOp {
    storage:         OperatorStorage,
    context:         CPUContext,

    ws:              *mut Workspace,

    /// QNNPACK convolution object
    qnnpack_object:  QnnpOperator, // default = nullptr

    /**
      | batch size in the previous call to
      | RunOnDeviceWithOrderNHWC
      |
      */
    last_batch_size: usize, // default = 0

    /**
      | input pointer in the previous call to
      | RunOnDeviceWithOrderNHWC
      |
      */
    last_input_pointer: *const c_void, // default = nullptr

    /**
      | output pointer in the previous call
      | to RunOnDeviceWithOrderNHWC
      |
      */
    last_output_pointer: *mut c_void, // default = nullptr
}

register_cpu_operator!{Int8FC, int8::Int8FCOp}

num_inputs!{Int8FC, (3,4)}

num_outputs!{Int8FC, (1,4)}

tensor_inference_function!{Int8FC, /* std::bind(FCShapeInference, _1, _2, false) */}

cost_inference_function!{Int8FC, /* std::bind(CostInferenceForFC, _1, _2, false) */ }

inputs!{Int8FC, 
    0 => ("X", "input tensor that's coerced into a 2D matrix of size (MxK) as described above"),
    1 => ("W", "A tensor that is coerced into a 2D blob of size (KxN) containing fully connected weight matrix"),
    2 => ("b", "1D blob containing bias vector"),
    3 => ("Qparam", "Optional Qparam blob that contains quant param computed on activation histogram data Will overwrite Y_scale and Y_zero_point argument if specified")
}

outputs!{Int8FC, 
    0 => ("Y", "2D output tensor")
}

args!{Int8FC, 
    0 => ("Y_scale",      "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

impl Drop for Int8FCOp {
    fn drop(&mut self) {
        todo!();
        /* 
        if (this->qnnpackObject_ != nullptr) {
          qnnp_delete_operator(this->qnnpackObject_);
          this->qnnpackObject_ = nullptr;
        }
       */
    }
}

impl Int8FCOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), ws_(ws) 
        createSharedBuffer<CPUContext>(ws_);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->Get<Int8TensorCPU>();
        const auto& W = Inputs()[1]->Get<Int8TensorCPU>();
        const auto& B = Inputs()[2]->Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;
        // (NxHxW)xC == MxK x (NxK) -> MxN
        const auto K = X.t.size_from_dim(1);
        const auto N = W.t.size(0);
        CHECK_EQ(K, W.t.size(1));
        CHECK_EQ(N, B.t.numel());
        const auto M = X.t.numel() / K;
        ReinitializeTensor(&Y->t, {M, N}, at::dtype<uint8_t>().device(CPU));

        runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
          initQNNPACK();

          pthreadpool_t threadpool =
              reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

          if (this->qnnpackObject_ == nullptr) {
            const qnnp_status createStatus = qnnp_create_fully_connected_nc_q8(
                K,
                N,
                X.zero_point,
                X.scale,
                W.zero_point,
                W.scale,
    #ifndef _MSC_VER
                W.t.template data<uint8_t>(),
                B.t.template data<int32_t>(),
    #else
                W.t.data<uint8_t>(),
                B.t.data<int32_t>(),
    #endif

                Y->zero_point,
                Y->scale,
                uint8_t::min,
                uint8_t::max,
                0 /* flags */,
                &this->qnnpackObject_);
            CAFFE_ENFORCE(
                createStatus == qnnp_status_success,
                "failed to create QNNPACK fully connected operator");
            CAFFE_ENFORCE(this->qnnpackObject_ != nullptr);
          }

          uint8_t* inputPtr = X.t.template mutable_data<uint8_t>();
          if (K < 8) {
            buffer->Resize(std::vector<int64_t>{X.t.numel() + 8});
            inputPtr = buffer->template mutable_data<uint8_t>() + 8;
            memcpy(inputPtr, X.t.template data<uint8_t>(), X.t.numel());
          }

          if (lastBatchSize_ != static_cast<size_t>(M) ||
              lastInputPointer_ != inputPtr ||
              lastOutputPointer_ != Y->t.template mutable_data<uint8_t>()) {
            const qnnp_status setupStatus = qnnp_setup_fully_connected_nc_q8(
                this->qnnpackObject_,
                M,
                inputPtr,
                K /* input stride */,
                Y->t.template mutable_data<uint8_t>(),
                N /* output stride */);
            CAFFE_ENFORCE(
                setupStatus == qnnp_status_success,
                "failed to setup QNNPACK fully connected operator");

            lastBatchSize_ = static_cast<size_t>(M);
            lastInputPointer_ = inputPtr;
            lastOutputPointer_ = Y->t.template mutable_data<uint8_t>();
          }

    #if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
          const qnnp_status runStatus =
              qnnp_run_operator(this->qnnpackObject_, nullptr /* thread pool */);
    #else
          const qnnp_status runStatus =
              qnnp_run_operator(this->qnnpackObject_, threadpool);
    #endif
          CAFFE_ENFORCE(
              runStatus == qnnp_status_success, "failed to run QNNPACK operator");
        });
        return true;
        */
    }
}
