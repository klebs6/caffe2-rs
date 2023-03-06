crate::ix!();

/**
  | LeakyRelu takes input data (Tensor<T>)
  | and an argument alpha, and produces
  | one output data (Tensor<T>) where the
  | function `f(x) = alpha x for x < 0`, `f(x)
  | = x for x >= 0`, is applied to the data tensor
  | elementwise.
  |
  */
pub struct Int8LeakyReluOp {
    storage: OperatorStorage,
    context: CPUContext,

    alpha: f32,
    ws:    *mut Workspace,

    /// QNNPACK Leaky ReLU operator
    qnnpack_operator: QnnpOperator, // default = nullptr
}

register_cpu_operator!{Int8LeakyRelu, int8::Int8LeakyReluOp}

num_inputs!{Int8LeakyRelu, 1}

num_outputs!{Int8LeakyRelu, 1}

inputs!{Int8LeakyRelu, 
    0 => ("X", "1D input tensor")
}

outputs!{Int8LeakyRelu, 
    0 => ("Y", "1D input tensor")
}

args!{Int8LeakyRelu, 
    0 => ("alpha",        "Coefficient of leakage, default value is 0.01"),
    1 => ("Y_scale",      "Output tensor quantization scale"),
    2 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape!{Int8LeakyRelu}

cost_inference_function!{
    Int8LeakyRelu, 
    PointwiseCostInference::<2> 
}

allow_inplace!{Int8LeakyRelu, vec![(0, 0)]}

impl Int8LeakyReluOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), ws_(ws) 

        const float alpha = this->template GetSingleArgument<float>("alpha", 0.01);
        CAFFE_ENFORCE_GT(alpha, 0.0);
        CAFFE_ENFORCE_LT(alpha, 1.0);
        this->alpha_ = alpha;
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
        CHECK_GE(Y_zero_point, uint8_t::min);
        CHECK_LE(Y_zero_point, uint8_t::max);

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
          const qnnp_status createStatus = qnnp_create_leaky_relu_nc_q8(
              1 /* channels */,
              this->alpha_,
              static_cast<uint8_t>(X_zero_point),
              X_scale,
              static_cast<uint8_t>(Y_zero_point),
              Y_scale,
              0 /* output min */,
              255 /* output max */,
              0 /* flags */,
              &qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK Leaky ReLU operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_leaky_relu_nc_q8(
            this->qnnpackOperator_,
            X.t.numel() /* batch size */,
            X.t.template data<uint8_t>(),
            1 /* X stride */,
            Y->t.template mutable_data<uint8_t>(),
            1 /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK Leaky ReLU operator");

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
            "failed to run QNNPACK Leaky ReLU operator");

        return true;
        */
    }
}

impl Drop for Int8LeakyReluOp {

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
