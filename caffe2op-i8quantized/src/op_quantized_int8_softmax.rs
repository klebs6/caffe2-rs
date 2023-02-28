crate::ix!();

/**
  | The operator computes the softmax normalized
  | values for each layer in the batch of
  | the given input.
  | 
  | The input is a 2-D tensor (Tensor<float>)
  | of size (batch_size x input_feature_dimensions).
  | 
  | The output tensor has the same shape
  | and contains the softmax normalized
  | values of the corresponding input.
  | 
  | X does not need to explicitly be a 2D vector;
  | rather, it will be coerced into one.
  | 
  | For an arbitrary n-dimensional tensor
  | 
  | X \in [a_0, a_1, ..., a_{k-1}, a_k, ...,
  | a_{n-1}] and k is the axis provided,
  | then X will be coerced into a 2-dimensional
  | tensor with dimensions [a_0 * ... * a_{k-1},
  | a_k * ... * a_{n-1}].
  | 
  | For the default case where axis=1, this
  | means the X tensor will be coerced into
  | a 2D tensor of dimensions [a_0, a_1 *
  | ... * a_{n-1}], where a_0 is often the
  | batch size.
  | 
  | In this situation, we must have a_0 =
  | N and a_1 * ... * a_{n-1} = D.
  | 
  | Each of these dimensions must be matched
  | correctly, or else the operator will
  | throw errors.
  |
  */
pub struct Int8SoftmaxOp {
    storage:          OperatorStorage,
    context:          CPUContext,
    ws:               *mut Workspace,

    /// QNNPACK SoftArgMax operator
    qnnpack_operator: QnnpOperator, // default = nullptr
}

register_cpu_operator!{Int8Softmax, int8::Int8SoftmaxOp}

num_inputs!{Int8Softmax, 1}

num_outputs!{Int8Softmax, 1}

inputs!{Int8Softmax, 
    0 => ("input", "The input tensor that's coerced into a 2D matrix of size (NxD) as described above.")
}

outputs!{Int8Softmax, 
    0 => ("output", "The softmax normalized output values with the same shape as input tensor.")
}

args!{Int8Softmax, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset"),
    2 => ("axis", "(int) default to 1; describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size")
}

identical_type_and_shape!{Int8Softmax}

impl Drop for Int8SoftmaxOp {
    fn drop(&mut self) {
        todo!();
        /* 
        if (this->qnnpackOperator_ != nullptr) {
          qnnp_delete_operator(this->qnnpackOperator_);
          this->qnnpackOperator_ = nullptr;
        }
       */
    }
}

impl Int8SoftmaxOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), ws_(ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        const int32_t Y_zero_point =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CHECK_EQ(Y_zero_point, 0);
        CHECK_EQ(Y_scale, 1.0f / 256.0f);

        /*
         * Record quantization parameters for the input, because if the op is
         * in-place, we may overwrite these parameters later, when we set
         * quantization parameters for output tensor.
         */
        const uint8_t X_zero_point = X.zero_point;
        const float X_scale = X.scale;

        Y->scale = Y_scale;
        Y->zero_point = Y_zero_point;
        Y->t.ResizeLike(X.t);

        initQNNPACK();

        if (this->qnnpackOperator_ == nullptr) {
          const qnnp_status createStatus = qnnp_create_softargmax_nc_q8(
              X.t.numel() / X.t.size(0) /* channels */,
              X_scale,
              static_cast<uint8_t>(Y_zero_point),
              Y_scale,
              0 /* flags */,
              &qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK SoftArgMax operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_softargmax_nc_q8(
            this->qnnpackOperator_,
            X.t.size(0) /* batch size */,
            X.t.template data<uint8_t>(),
            X.t.numel() / X.t.size(0) /* X stride */,
            Y->t.template mutable_data<uint8_t>(),
            X.t.numel() / X.t.size(0) /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK SoftArgMax operator");

    #if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
        const qnnp_status runStatus =
            qnnp_run_operator(this->qnnpackOperator_, nullptr /* thread pool */);
    #else
        pthreadpool_t threadpool =
            reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
        const qnnp_status runStatus =
            qnnp_run_operator(this->qnnpackOperator_, threadpool);
    #endif
        CAFFE_ENFORCE(
            runStatus == qnnp_status_success,
            "failed to run QNNPACK SoftArgMax operator");

        return true;
        */
    }
}

