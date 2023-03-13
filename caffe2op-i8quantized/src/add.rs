crate::ix!();

pub struct Int8AddOp<Activation> {
    storage:           OperatorStorage,
    context:           CPUContext,
    ws:                *mut Workspace,

    /// QNNPACK add operator
    qnnpack_operator:  QnnpOperator,
    phantomActivation: PhantomData<Activation>,
}

impl<Activation> Drop for Int8AddOp<Activation> {

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

impl<Activation> Int8AddOp<Activation> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), ws_(ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(Inputs().size(), 2);
        const auto& A = Inputs()[0]->template Get<Int8TensorCPU>();
        const auto& B = Inputs()[1]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();

        CAFFE_ENFORCE_EQ(
            A.t.sizes(),
            B.t.sizes(),
            "inputs must have the same shape (broadcast semantics is not supported)");

        /*
         * Record quantization parameters for A and B inputs, because if the op is
         * in-place, we may overwrite these parameters later, when we set
         * quantization parameters for Y tensor.
         */
        const uint8_t A_zero_point = A.zero_point;
        const uint8_t B_zero_point = B.zero_point;
        const float A_scale = A.scale;
        const float B_scale = B.scale;

        const int32_t Y_zero_point =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->t.ResizeLike(A.t);
        Y->zero_point = Y_zero_point;
        Y->scale = Y_scale;

        initQNNPACK();

        pthreadpool_t threadpool =
            reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

        if (this->qnnpackOperator_ == nullptr) {
          const qnnp_status createStatus = qnnp_create_add_nc_q8(
              1 /* channels */,
              A_zero_point,
              A_scale,
              B_zero_point,
              B_scale,
              static_cast<uint8_t>(Y_zero_point),
              Y_scale,
              activationLimits(Y_scale, Y_zero_point, Ac).first,
              activationLimits(Y_scale, Y_zero_point, Ac).second,
              0 /* flags */,
              &qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK add operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_add_nc_q8(
            this->qnnpackOperator_,
            A.t.numel() /* batch size */,
            A.t.template data<uint8_t>(),
            1 /* A stride */,
            B.t.template data<uint8_t>(),
            1 /* B stride */,
            Y->t.template mutable_data<uint8_t>(),
            1 /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK add operator");

    #if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
        const qnnp_status runStatus =
            qnnp_run_operator(this->qnnpackOperator_, nullptr /* thread pool */);
    #else
        const qnnp_status runStatus =
            qnnp_run_operator(this->qnnpackOperator_, threadpool);
    #endif
        CAFFE_ENFORCE(
            runStatus == qnnp_status_success, "failed to run QNNPACK add operator");

        return true;
        */
    }
}

/**
  | Performs element-wise binary Add (with
  | no broadcast support).
  |
  */
register_cpu_operator!{Int8Add,     Int8AddOp<Activation::NONE>}

num_inputs!{Int8Add, 2}

num_outputs!{Int8Add, 1}

inputs!{Int8Add, 
    0 => ("A", "First operand, should share the type with the second operand."),
    1 => ("B", "Second operand. It should be of the same size as A.")
}

outputs!{Int8Add, 
    0 => ("C", "Result, has same dimensions and type as A")
}

args!{Int8Add, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

allow_inplace!{Int8Add, vec![(0, 0), (1, 0)]}

/**
  | Performs element-wise binary Add (with
  | no broadcast support).
  | 
  | Output will go through rectified linear
  | function, where y = max(0, x).
  |
  */
register_cpu_operator!{Int8AddRelu, Int8AddOp<Activation::RELU>}

num_inputs!{Int8AddRelu, 2}

num_outputs!{Int8AddRelu, 1}

inputs!{Int8AddRelu, 
    0 => ("A", "First operand, should share the type with the second operand."),
    1 => ("B", "Second operand. It should be of the same size as A.")
}

outputs!{Int8AddRelu, 
    0 => ("C", "Result, has same dimensions and type as A")
}

args!{Int8AddRelu, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

allow_inplace!{Int8AddRelu, vec![(0, 0), (1, 0)]}

/**
  | These ops are defined as alias of
  | 
  | Int8Add/Int8AddRelu for compatibility
  | with current production models. In
  | the future these ops will be changed
  | to an equivalent of Sum op, which does
  | reduction of a single argument.
  | 
  | We deliberately omit schema for
  | Int8Sum/Int8SumRelu so they can temporary
  | use either legacy or the new semantics
  | depending on the engine.
  |
  */
register_cpu_operator!{Int8Sum,     Int8AddOp<Activation::NONE>}

num_inputs!{Int8Sum, (1, i32::MAX)}

num_outputs!{Int8Sum, 1}

args!{Int8Sum, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape_of_input!{Int8Sum, 0}

allow_inplace!{Int8Sum, vec![(0, 0), (1, 0)]}

cost_inference_function!{Int8Sum, CostInferenceForSum}

///------------------------------------
register_cpu_operator!{Int8SumRelu, Int8AddOp<Activation::RELU>}

num_inputs!{Int8Sum, (1, i32::MAX)}

num_outputs!{Int8SumRelu, 1}

args!{Int8SumRelu, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape_of_input!{Int8SumRelu, 0}

cost_inference_function!{Int8SumRelu, CostInferenceForSum}

allow_inplace!{Int8SumRelu, vec![(0, 0), (1, 0)]}
