crate::ix!();

/**
  | The transposed convolution consumes
  | an input vector, the filter blob, and
  | the bias blob, and computes the output.
  | 
  | -----------
  | @note
  | 
  | other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | ConvTransposeUnpoolOpBase operator.
  | 
  | Various dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator.
  | 
  | As is expected, the filter is deconvolved
  | with a subset of the image and the bias
  | is added; this is done throughout the
  | image data and the output is computed.
  | 
  | As a side note on the implementation
  | layout: conv_transpose_op_impl.h
  | is the templated implementation of
  | the conv_transpose_op.h file, which
  | is why they are separate files.
  |
  */
pub struct Int8ConvTransposeOp {
    
    //USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(CPUContext);
    base: ConvTransposeUnpoolBase<CPUContext>,

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

register_cpu_operator!{Int8ConvTranspose, int8::Int8ConvTransposeOp}

num_inputs!{Int8ConvTranspose, (2,3)}

num_outputs!{Int8ConvTranspose, 1}

inputs!{Int8ConvTranspose, 
    0 => ("X",            "Input data blob from previous layer; has size (N x H x W x C), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that NHWC is supported now"),
    1 => ("filter",       "The filter blob that will be used in the transposed convolution; has size (M x kH x kW x C), where C is the number of channels, and kH and kW are the height and width of the kernel."),
    2 => ("bias",         "The 1D bias blob that is added through the convolution; has size (C). Optional, if not passed, will treat it as all 0.")
}

outputs!{Int8ConvTranspose, 
    0 => ("Y",            "Output data blob that contains the result of the transposed convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

args!{Int8ConvTranspose, 
    0 => ("Y_scale",      "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

impl Drop for Int8ConvTransposeOp {
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

impl Int8ConvTransposeOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvTransposeUnpoolBase(std::forward<Args>(args)...) 

        OPERATOR_NEEDS_FEATURE(
            this->order_ == StorageOrder::NHWC,
            "Int8ConvTransposeOp only supports NHWC order");
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
        const auto X_offset = -X.zero_point;
        const auto W_offset = -W.zero_point;
        const int32_t Y_offset =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        double Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;

        const auto N = X.t.size(0);
        const auto IH = X.t.size(1);
        const auto IW = X.t.size(2);
        const auto IC = X.t.size(3);

        CHECK_EQ(IC, W.t.size(0));
        const auto KH = W.t.size(1);
        const auto KW = W.t.size(2);
        const auto OC = W.t.size(3);

        auto sizes = ConvTransposeUnpoolBase<CPUContext>::GetOutputSize(X.t, OC);
        ReinitializeTensor(&(Y->t), sizes, at::dtype<uint8_t>().device(CPU));
        CHECK_EQ(OC, Y->t.size(3));

        runWithSharedBuffer<CPUContext>(ws_, [&](Tensor* buffer) {
          initQNNPACK();

          pthreadpool_t threadpool =
              reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());

          if (this->qnnpackObject_ == nullptr) {
            const qnnp_status createStatus = qnnp_create_deconvolution2d_nhwc_q8(
                pad_t(),
                pad_r(),
                pad_b(),
                pad_l(),
                adj_h(),
                adj_w(),
                KH,
                KW,
                stride_h(),
                stride_w(),
                1 /* dilation height */,
                1 /* dilation width */,
                1 /* groups */,
                IC,
                OC,
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
                "failed to create QNNPACK convolution object");
            CAFFE_ENFORCE(this->qnnpackObject_ != nullptr);
          }

          uint8_t* inputPtr = X.t.template mutable_data<uint8_t>();
          if (IC < 8) {
            buffer->Resize(std::vector<int64_t>{X.t.numel() + 8});
            inputPtr = buffer->template mutable_data<uint8_t>() + 8;
            memcpy(inputPtr, X.t.template data<uint8_t>(), X.t.numel());
          }

          if (lastBatchSize_ != static_cast<size_t>(X.t.size(0)) ||
              lastInputHeight_ != static_cast<size_t>(X.t.size(1)) ||
              lastInputWidth_ != static_cast<size_t>(X.t.size(2)) ||
              lastInputPointer_ != inputPtr ||
              lastOutputPointer_ != Y->t.template mutable_data<uint8_t>()) {
            const qnnp_status setupStatus = qnnp_setup_deconvolution2d_nhwc_q8(
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
