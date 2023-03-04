crate::ix!();

pub struct Int8ChannelShuffleOp {
    base: ConvPoolOpBase<CPUContext>,

    ws: *mut Workspace,

    /// QNNPACK channel shuffle operator
    qnnpack_operator: QnnpOperator,
}

impl Drop for Int8ChannelShuffleOp {
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

impl Int8ChannelShuffleOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CPUContext>(operator_def, ws), ws_(ws) 

        OPERATOR_NEEDS_FEATURE(
            this->order_ == StorageOrder::NHWC,
            "Int8ChannelShuffleOp only supports NHWC order");
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        Y->t.ResizeLike(X.t);
        Y->scale = X.scale;
        Y->zero_point = X.zero_point;
        const int32_t Y_offset =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        const float Y_scale =
            this->template GetSingleArgument<float>("Y_scale", 1.0f);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);
        CHECK_GE(X.zero_point, uint8_t::min);
        CHECK_LE(X.zero_point, uint8_t::max);

        const auto C = X.t.dim32(3);
        const auto G = this->group_;
        CAFFE_ENFORCE(C % G == 0, "");
        const auto B = X.t.numel() / C;

        initQNNPACK();

        if (this->qnnpackOperator_ == nullptr) {
          const qnnp_status createStatus = qnnp_create_channel_shuffle_nc_x8(
              G /* groups */,
              C / G /* group channels */,
              0 /* flags */,
              &this->qnnpackOperator_);
          CAFFE_ENFORCE(
              createStatus == qnnp_status_success,
              "failed to create QNNPACK channel shuffle operator");
          CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
        }

        const qnnp_status setupStatus = qnnp_setup_channel_shuffle_nc_x8(
            this->qnnpackOperator_,
            X.t.numel() / C /* batch size */,
            X.t.template data<uint8_t>(),
            C /* X stride */,
            Y->t.template mutable_data<uint8_t>(),
            C /* Y stride */);
        CAFFE_ENFORCE(
            setupStatus == qnnp_status_success,
            "failed to setup QNNPACK channel shuffle operator");

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
            "failed to run QNNPACK channel shuffle operator");

        return true;
        */
    }
}

register_cpu_operator!{Int8ChannelShuffle, Int8ChannelShuffleOp}

num_inputs!{Int8ChannelShuffle, 1}

num_outputs!{Int8ChannelShuffle, 1}

args!{Int8ChannelShuffle, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

identical_type_and_shape!{Int8ChannelShuffle}
