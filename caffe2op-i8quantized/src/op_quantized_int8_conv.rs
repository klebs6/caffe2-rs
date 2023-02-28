crate::ix!();

/**
  | The convolution operator consumes
  | an input vector, a ND filter blob and
  | a bias blob and computes the output.
  | 
  | [Only NHWC order is supported now]Note
  | that other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | ConvPoolOpBase operator.
  | 
  | Various dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator.
  | 
  | As is expected, the filter is convolved
  | with a subset of the image and the bias
  | is added; this is done throughout the
  | image data and the output is computed.
  | 
  | As a side note on the implementation
  | layout: conv_op_impl.h is the templated
  | implementation of the conv_op.h file,
  | which is why they are separate files.
  |
  */
pub struct Int8ConvOp<Activation> {

    //USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
    base: ConvPoolOpBase<CPUContext>,

    /// QNNPACK convolution object
    qnnpack_object: QnnpOperator, // default = nullptr

    /**
      | batch size in the previous call to
      | RunOnDeviceWithOrderNHWC
      |
      */
    last_batch_size: usize, // default = 0

    /**
      | input height in the previous call to
      | RunOnDeviceWithOrderNHWC
      |
      */
    last_input_height: usize, // default = 0

    /**
      | input width in the previous call to
      | RunOnDeviceWithOrderNHWC
      |
      */
    last_input_width: usize, // default = 0

    /**
      | input pointer in the previous call to
      | 
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
    phantomActivation:   PhantomData<Activation>,
}

impl Int8ConvOp<Activation> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase(std::forward<Args>(args)...) 

        OPERATOR_NEEDS_FEATURE(
            this->order_ == StorageOrder::NHWC,
            "Int8Conv only supports NHWC order");
        createSharedBuffer<CPUContext>(ws_);
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(Inputs().size(), 3);
        const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
        const auto& W = Inputs()[1]->template Get<Int8TensorCPU>();
        const auto& B = Inputs()[2]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        const int32_t Y_offset =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        double Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);

        ConvPoolOpBase<CPUContext>::SetOutputSize(X.t, &(Y->t), W.t.dim32(0));
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;

        const auto M = W.t.size(0);
        const auto KH = W.t.size(1);
        const auto KW = W.t.size(2);
        const auto KC = W.t.size(3);
        const auto C = X.t.dim32(3);
        const bool isDepthwise = this->group_ > 1 && this->group_ == M &&
            this->group_ == C && KC == 1 && KH * KW == 9 && dilation_w() == 1;

        CHECK_EQ(Y->t.dim32(3), M);
        runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
          initQNNPACK();

          pthreadpool_t threadpool =
              reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

          if (this->qnnpackObject_ == nullptr) {
            CAFFE_ENFORCE(
                C % this->group_ == 0,
                "number of input channels must be divisible by groups count");
            CAFFE_ENFORCE(
                M % this->group_ == 0,
                "number of output channels must be divisible by groups count");
            const qnnp_status createStatus = qnnp_create_convolution2d_nhwc_q8(
                pad_t(),
                pad_r(),
                pad_b(),
                pad_l(),
                KH,
                KW,
                stride_h(),
                stride_w(),
                dilation_h(),
                dilation_w(),
                this->group_,
                C / this->group_,
                M / this->group_,
                X.zero_point,
                X.scale,
                W.zero_point,
                W.scale,
    #if !defined(_MSC_VER) || defined(__clang__)
                W.t.template data<uint8_t>(),
                B.t.template data<int32_t>(),
    #else
                W.t.data<uint8_t>(),
                B.t.data<int32_t>(),
    #endif
                Y->zero_point,
                Y->scale,
                activationLimits(Y->scale, Y->zero_point, Ac).first,
                activationLimits(Y->scale, Y->zero_point, Ac).second,
                0 /* flags */,
                &this->qnnpackObject_);
            CAFFE_ENFORCE(
                createStatus == qnnp_status_success,
                "failed to create QNNPACK convolution object");
            CAFFE_ENFORCE(this->qnnpackObject_ != nullptr);
          }

          uint8_t* inputPtr = X.t.template mutable_data<uint8_t>();
          if ((isDepthwise && this->group_ < 8) ||
              (!isDepthwise && C / this->group_ < 8)) {
            buffer->Resize(std::vector<int64_t>{X.t.numel() + 8});
            inputPtr = buffer->template mutable_data<uint8_t>() + 8;
            memcpy(inputPtr, X.t.template data<uint8_t>(), X.t.numel());
          }

          if (lastBatchSize_ != static_cast<size_t>(X.t.size(0)) ||
              lastInputHeight_ != static_cast<size_t>(X.t.size(1)) ||
              lastInputWidth_ != static_cast<size_t>(X.t.size(2)) ||
              lastInputPointer_ != inputPtr ||
              lastOutputPointer_ != Y->t.template mutable_data<uint8_t>()) {
            const qnnp_status setupStatus = qnnp_setup_convolution2d_nhwc_q8(
                this->qnnpackObject_,
                X.t.size(0),
                X.t.size(1),
                X.t.size(2),
                inputPtr,
                X.t.size(3) /* input pixel stride */,
                Y->t.template mutable_data<uint8_t>(),
                Y->t.size(3) /* output pixel stride */,
                nullptr /* threadpool */);
            CAFFE_ENFORCE(
                setupStatus == qnnp_status_success,
                "failed to setup QNNPACK convolution object");

            lastBatchSize_ = static_cast<size_t>(X.t.size(0));
            lastInputHeight_ = static_cast<size_t>(X.t.size(1));
            lastInputWidth_ = static_cast<size_t>(X.t.size(2));
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
              runStatus == qnnp_status_success,
              "failed to run QNNPACK convolution");
        });
        return true;
        */
    }
}

impl<Activation> Drop for Int8ConvOp<Activation> {
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

///-------------------------------------
register_cpu_operator!{Int8Conv, Int8ConvOp<Activation::NONE>}

num_inputs!{Int8Conv, (2,3)}

num_outputs!{Int8Conv, 1}

inputs!{Int8Conv, 
    0 => ("X", "Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. "),
    1 => ("filter", "The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel."),
    2 => ("bias", "The 1D bias blob that is added through the convolution; has size (M).")
}

outputs!{Int8ConvRelu, 
    0 => ("Y", "Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

args!{Int8Conv, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{Int8Conv, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}

cost_inference_function!{
    Int8Conv, 
    OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase::<CPUContext>::CostInferenceForConv
    )
}

///-------------------------------------
register_cpu_operator!{Int8ConvRelu, Int8ConvOp::<Activation::RELU>}

num_inputs!{Int8ConvRelu, (2,3)}

num_outputs!{Int8ConvRelu, 1}

inputs!{Int8ConvRelu, 
    0 => ("X", "Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. "),
    1 => ("filter", "The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel."),
    2 => ("bias", "The 1D bias blob that is added through the convolution; has size (M).")
}

outputs!{Int8ConvRelu, 
    0 => ("Y", 
        "Output data blob that contains the result of the convolution. 
        The output dimensions are functions of the kernel size, stride size, and pad lengths. 
        Output will go through rectified linear function, where y = max(0, x).")
}

args!{Int8ConvRelu, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{
    Int8ConvRelu, 
    ConvPoolOpBase::<CPUContext>::TensorInferenceForConv
}

cost_inference_function!{Int8ConvRelu, 
    OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase::<CPUContext>::CostInferenceForConv)
}
