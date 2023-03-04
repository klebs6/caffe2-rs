crate::ix!();

/**
  | Apply the Sigmoid function element-wise
  | to the input tensor.
  | 
  | This is often used as a non-linear activation
  | function in a neural network.
  | 
  | The sigmoid function is defined as:
  | 
  | $$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc
  |
  */
pub struct Int8SigmoidOp {
    storage: OperatorStorage,
    context: CPUContext,
    ws:      *mut Workspace,

    /// QNNPACK Sigmoid operator
    qnnpack_operator: QnnpOperator, // default = nullptr
}

register_cpu_operator!{Int8Sigmoid, int8::Int8SigmoidOp}

num_inputs!{Int8Sigmoid, 1}

num_outputs!{Int8Sigmoid, 1}

inputs!{Int8Sigmoid, 
    0 => ("input", "The input tensor that's coerced into a 2D matrix of size (NxD) as described above.")
}

outputs!{Int8Sigmoid, 
    0 => ("output", "The sigmoid normalized output values with the same shape as input tensor.")
}

args!{Int8Sigmoid, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape!{Int8Sigmoid}

impl Drop for Int8SigmoidOp {
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

impl Int8SigmoidOp {
    
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
          const qnnp_status createStatus = qnnp_create_sigmoid_nc_q8(
              1 /* channels */,
              X_zero_point,
              X_scale,
              static_cast<uint8_t>(Y_zero_point),
              Y_scale,
              0 /* output min */,
              255 /* output max */,
              0 /* flags */,
              &qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK Sigmoid operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_sigmoid_nc_q8(
            this->qnnpackOperator_,
            X.t.numel() /* batch size */,
            X.t.template data<uint8_t>(),
            1 /* X stride */,
            Y->t.template mutable_data<uint8_t>(),
            1 /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK Sigmoid operator");

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
            "failed to run QNNPACK Sigmoid operator");

        return true;
        */
    }
}
