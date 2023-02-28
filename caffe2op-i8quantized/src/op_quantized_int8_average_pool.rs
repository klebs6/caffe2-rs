crate::ix!();

/**
  | consumes an input blob X and applies
  | average pooling across the the blob
  | according to kernel sizes, stride sizes,
  | and pad lengths defined by the ConvPoolOpBase
  | operator.
  | 
  | Average pooling consisting of averaging
  | all values of a subset of the input tensor
  | according to the kernel size and downsampling
  | the data into the output blob Y for further
  | processing.
  |
  */
pub struct Int8AveragePoolOp<Activation> {

    base: ConvPoolOpBase<CPUContext>,

    /// QNNPACK Average Pooling operator
    qnnpack_operator:        QnnpOperator,

    /// QNNPACK Global Average Pooling operator
    qnnpack_global_operator: QnnpOperator,
    phantomActivation:       PhantomData<Activation>,
}

impl<Activation> Drop for Int8AveragePoolOp<Activation> {

    fn drop(&mut self) {

        todo!();

        /*
        if (this->qnnpackOperator_ != nullptr) {
          qnnp_delete_operator(this->qnnpackOperator_);
          this->qnnpackOperator_ = nullptr;
        }
        if (this->qnnpackGlobalOperator_ != nullptr) {
          qnnp_delete_operator(this->qnnpackGlobalOperator_);
          this->qnnpackGlobalOperator_ = nullptr;
        }
        */
    }
}

impl<Activation> Int8AveragePoolOp<Activation> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CPUContext>(std::forward<Args>(args)...) 

        OPERATOR_NEEDS_FEATURE(
            this->order_ == StorageOrder::NHWC, "Int8 only supports NHWC order.");
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        int32_t Y_zero_point =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->scale = Y_scale;
        Y->zero_point = Y_zero_point;

        CHECK_EQ(X.t.dim(), 4);
        const int channels = X.t.dim32(3);
        ConvPoolOpBase<CPUContext>::SetOutputSize(X.t, &(Y->t), channels);

        initQNNPACK();

        const bool anyPadding =
            pad_t() != 0 || pad_r() != 0 || pad_b() != 0 || pad_l() != 0;
        const bool anyStride = stride_h() > 1 || stride_w() > 1;
        const bool globalPooling = !anyPadding && !anyStride &&
            (X.t.dim32(1) == kernel_h() && X.t.dim32(2) == kernel_w());
        if (globalPooling) {
          if (this->qnnpackGlobalOperator_ == nullptr) {
            const qnnp_status createStatus =
                qnnp_create_global_average_pooling_nwc_q8(
                    channels,
                    X.zero_point,
                    X.scale,
                    Y->zero_point,
                    Y->scale,
                    activationLimits(Y->scale, Y->zero_point, Ac).first,
                    activationLimits(Y->scale, Y->zero_point, Ac).second,
                    0 /* flags */,
                    &this->qnnpackGlobalOperator_);
            CAFFE_ENFORCE(
                createStatus == qnnp_status_success,
                "failed to create QNNPACK Global Average Pooling operator");
            CAFFE_ENFORCE(this->qnnpackGlobalOperator_ != nullptr);
          }

          const qnnp_status setupStatus = qnnp_setup_global_average_pooling_nwc_q8(
              this->qnnpackGlobalOperator_,
              X.t.dim32(0),
              X.t.dim32(1) * X.t.dim32(2),
              X.t.template data<uint8_t>(),
              channels,
              Y->t.template mutable_data<uint8_t>(),
              channels);
          CAFFE_ENFORCE(
              setupStatus == qnnp_status_success,
              "failed to setup QNNPACK Global Average Pooling operator");

    #if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
          const qnnp_status runStatus = qnnp_run_operator(
              this->qnnpackGlobalOperator_, nullptr /* thread pool */);
    #else
          pthreadpool_t threadpool =
              reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
          const qnnp_status runStatus =
              qnnp_run_operator(this->qnnpackGlobalOperator_, threadpool);
    #endif
          CAFFE_ENFORCE(
              runStatus == qnnp_status_success,
              "failed to run QNNPACK Global Average Pooling operator");
        } else {
          if (this->qnnpackOperator_ == nullptr) {
            const qnnp_status createStatus = qnnp_create_average_pooling2d_nhwc_q8(
                pad_t(),
                pad_r(),
                pad_b(),
                pad_l(),
                kernel_h(),
                kernel_w(),
                stride_h(),
                stride_w(),
                channels,
                X.zero_point,
                X.scale,
                Y->zero_point,
                Y->scale,
                activationLimits(Y->scale, Y->zero_point, Ac).first,
                activationLimits(Y->scale, Y->zero_point, Ac).second,
                0 /* flags */,
                &this->qnnpackOperator_);
            CAFFE_ENFORCE(
                createStatus == qnnp_status_success,
                "failed to create QNNPACK Average Pooling operator");
            CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
          }

          const qnnp_status setupStatus = qnnp_setup_average_pooling2d_nhwc_q8(
              this->qnnpackOperator_,
              X.t.dim32(0),
              X.t.dim32(1),
              X.t.dim32(2),
              X.t.template data<uint8_t>(),
              channels,
              Y->t.template mutable_data<uint8_t>(),
              channels,
              nullptr /* thread pool */);
          CAFFE_ENFORCE(
              setupStatus == qnnp_status_success,
              "failed to setup QNNPACK Average Pooling operator");

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
              "failed to run QNNPACK Average Pooling operator");
        }

        return true;
        */
    }
}

///--------------------
register_cpu_operator!{Int8AveragePool, Int8AveragePoolOp<Activation::NONE>}

num_inputs!{Int8AveragePool, 1}

num_outputs!{Int8AveragePool, 1}

inputs!{Int8AveragePool, 
    0 => ("X", "Input data tensor from the previous operator; dimensions depend on whether the NCHW or 
        NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), 
        where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. 
        The corresponding permutation of dimensions is used in the latter case.")
}

outputs!{Int8AveragePool, 
    0 => ("Y", "Output data tensor from average pooling across the input tensor. 
        Dimensions will vary based on various kernel, stride, and pad sizes.")
}

args!{Int8AveragePool, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{
    Int8AveragePool, 
    ConvPoolOpBase::<CPUContext>::TensorInferenceForPool
}

///--------------------
register_cpu_operator!{
    Int8AveragePoolRelu, 
    Int8AveragePoolOp::<Activation::RELU>
}

num_inputs!{Int8AveragePoolRelu, 1}

num_outputs!{Int8AveragePoolRelu, 1}

args!{Int8AveragePoolRelu, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

inputs!{Int8AveragePoolRelu, 
    0 => ("X", "Input data tensor from the previous operator; dimensions depend on whether the NCHW or 
        NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), 
        where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. 
        The corresponding permutation of dimensions is used in the latter case.")
}

outputs!{Int8AveragePoolRelu, 
    0 => ("Y", "Output data tensor from average pooling across the input tensor. 
    Dimensions will vary based on various kernel, stride, and pad sizes. 
    Output will go through rectified linear function, where y = max(0, x)." )
}

tensor_inference_function!{
    Int8AveragePoolRelu,
    ConvPoolOpBase::<CPUContext>::TensorInferenceForPool
}
