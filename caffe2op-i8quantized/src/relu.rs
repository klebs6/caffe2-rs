crate::ix!();

/**
  | Relu takes one input data (Tensor<T>)
  | and produces one output data (Tensor<T>)
  | where the rectified linear function,
  | y = max(0, x), is applied to the tensor
  | elementwise.
  |
  */
pub struct Int8ReluOp {
    storage: OperatorStorage,
    context: CPUContext,

    ws: *mut Workspace,

    /// QNNPACK Clamp operator
    qnnpack_operator: QnnpOperator, // default = nullptr

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

register_cpu_operator!{Int8Relu, int8::Int8ReluOp}

num_inputs!{Int8Relu, 1}

num_outputs!{Int8Relu, 1}

inputs!{Int8Relu, 
    0 => ("X", "1D input tensor")
}

outputs!{Int8Relu, 
    0 => ("Y", "1D input tensor")
}

args!{Int8Relu, 
    0 => ("Y_scale",      "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape!{Int8Relu}

cost_inference_function!{Int8Relu, CostInferenceForRelu }

inherit_onnx_schema!{Int8Relu, "Relu"}

allow_inplace!{Int8Relu, vec![(0, 0)]}

impl Drop for Int8ReluOp {
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

impl Int8ReluOp {
    
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
        Y->t.ResizeLike(X.t);
        Y->scale = X.scale;
        Y->zero_point = X.zero_point;
        CHECK_GE(X.zero_point, uint8_t::min);
        CHECK_LE(X.zero_point, uint8_t::max);
        const int32_t Y_offset =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        const float Y_scale =
            this->template GetSingleArgument<float>("Y_scale", 1.0f);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);

        initQNNPACK();

        if (this->qnnpackOperator_ == nullptr) {
          const qnnp_status createStatus = qnnp_create_clamp_nc_u8(
              1 /* channels */,
              X.zero_point /* output min */,
              255 /* output max */,
              0 /* flags */,
              &qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK Clamp operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_clamp_nc_u8(
            this->qnnpackOperator_,
            X.t.numel() /* batch size */,
            X.t.template data<uint8_t>(),
            1 /* X stride */,
            Y->t.template mutable_data<uint8_t>(),
            1 /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK Clamp operator");

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
            "failed to run QNNPACK Clamp operator");

        return true;
        */
    }
}

#[inline] pub fn cost_inference_for_relu(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}
