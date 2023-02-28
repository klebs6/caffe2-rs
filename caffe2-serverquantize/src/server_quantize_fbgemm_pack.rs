crate::ix!();

pub type FCFp32Op = FullyConnectedOp<CPUContext,DefaultEngine,FCDefaultTransposeWeight>;

pub type ConvFp32Op = ConvOp<f32,CPUContext>;

///--------------------
pub struct FullyConnectedDNNLowPPackWeightOp {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    base: DNNLowPOp<u8, FCFp32Op>,

    axis_w:                 i32,
    quantize_channelwise:   bool,

    /// only for DNNLOWP_ACC16
    nbits_in_non_outlier:   i32,

    save_unpacked_weights:  bool,
}

input_tags!{
    FullyConnectedDNNLowPPackWeightOp
    {
        Filter,
        Bias
    }
}

impl FullyConnectedDNNLowPPackWeightOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : DNNLowPOp<uint8_t, FCFp32Op>(operator_def, ws),
        axis_w_(this->GetSingleArgument<int32_t>("axis_w", 1)),
        quantize_channelwise_(
            this->GetSingleArgument<bool>("quantize_channelwise", false)),
            save_unpacked_weights_(
                this->GetSingleArgument<bool>("save_unpacked_weights", false)) 

                if (this->debug_def().engine() == "DNNLOWP_ROWWISE") {
                    quantize_channelwise_ = true;
                }
        if (this->debug_def().engine() == "DNNLOWP_ACC16") {
            nbits_in_non_outlier_ = this->GetSingleArgument<int>(
                "nbits_in_non_outlier", FLAGS_caffe2_dnnlowp_nbits_in_non_outlier);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& filter = InputTensorCPU_(0);
        const auto canonical_axis_w = filter.canonical_axis_index(axis_w_);
        const auto K = filter.size_from_dim(canonical_axis_w);
        const auto N = filter.size_to_dim(canonical_axis_w);

        auto* Y = this->Output<Int8FCDNNLowPPackedWeightBlob>(0);

        // Create tensor with the same shape but this new tensor shouldn't actually
        // allocate memory for the tensor.
        // This is just a convenient way to pass tensor shape information
        Y->original_tensor.ResizeLike(filter);

        Y->qparams.resize(quantize_channelwise_ ? N : 1);

        vector<int8_t> W_quantized;
        QuantizeWeight<uint8_t>(
            InputBlob(0), K, N, Y->qparams, W_quantized, qfactory_.get());

        if (save_unpacked_weights_) {
            ReinitializeTensor(
                &Y->original_tensor, filter.sizes(), at::dtype<int8_t>().device(CPU));
            auto* buffer = Y->original_tensor.template mutable_data<int8_t>();
            CAFFE_ENFORCE_EQ(Y->original_tensor.numel(), W_quantized.size());
            memcpy(buffer, W_quantized.data(), W_quantized.size() * sizeof(int8_t));
        }
        if (this->InputIsType<int8::Int8TensorCPU>(0) && quantize_channelwise_) {
            static int log_occurences = 0;
            if (log_occurences < 32) {
                ++log_occurences;
                LOG(WARNING) << "Cannot do row-wise quantization for "
                    "pre-quantized weight "
                    << this->debug_def().input(0);
            }
        }

        // Pre-compute column offsets
        // This should happen before ExtractOutlierMatrix because W_quantized is
        // changed in ExtractOutlierMatrix.
        Y->column_offsets.reset(new vector<int32_t>());
        ComputeColumnOffsets(
            K, N, W_quantized.data(), Y->qparams, *Y->column_offsets);

        if (this->debug_def().engine() == "DNNLOWP_ACC16") {
            if (nbits_in_non_outlier_ < 8) {
                Y->W_outlier.reset(
                    ExtractOutlierMatrix(1, K, N, nbits_in_non_outlier_, W_quantized));
                int outlier_cnt = Y->W_outlier->ColPtr()[N];

                LOG(INFO) << "Proportion of outlier for FC layer with weight blob "
                    << this->debug_def().input(0) << " is "
                    << static_cast<float>(outlier_cnt) / W_quantized.size();
                LOG(INFO) << "nbits_in_non_outlier " << nbits_in_non_outlier_;
            }

            Y->nbits_in_non_outlier = nbits_in_non_outlier_;
            Y->W_acc16.reset(new fbgemm::PackBMatrix<int8_t, int16_t>(
                    fbgemm::matrix_op_t::Transpose,
                    K,
                    N,
                    W_quantized.data(),
                    K,
                    nullptr, // pmat
                    1)); // group
        } else {
            Y->W.reset(new fbgemm::PackBMatrix<int8_t>(
                    fbgemm::matrix_op_t::Transpose,
                    K,
                    N,
                    W_quantized.data(),
                    K,
                    nullptr, // pmat
                    1)); // group
        }

        // Quantize bias
        if (InputSize() >= 2) {
            TensorQuantizationParams in_qparams;
            CAFFE_ENFORCE(HasSingleArgumentOfType<float>("in_scale"));
            in_qparams.scale = GetSingleArgument<float>("in_scale", 0);
            Y->bias.reset(new vector<int32_t>());
            QuantizeConvBias(InputBlob(1), N, in_qparams, Y->qparams, *Y->bias);
        } else {
            Y->bias = nullptr;
        }

        // Output quantized bias if we specify a second output. This output is meant
        // to be consumed by accelerator instead of CPU ops.
        if (OutputSize() >= 2) {
            CAFFE_ENFORCE(Y->bias, "Bias is not quantized");
            // The reason we don't support this is basically due to limitation of
            // Int8TensorCPU only support single scale and zero_point. If we choose to
            // output bias as Int8FCDNNLowPPackedWeightBlob with original layout,
            // everything should still work for accelerator.
            CAFFE_ENFORCE_EQ(
                1,
                Y->qparams.size(),
                "We don't support outputing channelwise quantized bias yet");
            auto quantized_bias = Y->bias;
            float in_scale = GetSingleArgument<float>("in_scale", 0);
            float bias_scale = in_scale * Y->qparams.front().scale;
            LOG(INFO) << "Bias scale " << bias_scale << ": input scale " << in_scale
                << " weight scale " << Y->qparams.front().scale;
            auto* Bq = this->Output<int8::Int8TensorCPU>(1);
            std::vector<int64_t> shape = {static_cast<int64_t>(quantized_bias->size())};
            Bq->t.Resize(shape);
            Bq->scale = bias_scale;
            Bq->zero_point = 0;
            auto* data = Bq->t.template mutable_data<int32_t>();
            context_.template CopySameDevice<int32_t>(
                quantized_bias->size(), quantized_bias->data(), data);
        }

        return true;
        */
    }
}

/**
  | Pack a weight matrix that can be used
  | by DNNLOWP Int8Conv operators. DNNLOWP
  | operators can pack matrix on demand
  | during their first invocations but
  | calling this operator to pre-pack can
  | have benefits like saving memory space
  | when multiple operators are sharing
  | the same weight. This operator should
  | be a part of init net to be called once
  | to populate packed blob to be used by
  | Int8Conv DNNLOWP operators in the predictor
  | net
  | 
  | This operator optionally can also pre-quantize
  | bias. Then, we should also provide the
  | scale of input activation tensor as
  | in_scale argument.
  |
  */
pub struct ConvDNNLowPPackWeightOp {
    //USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
    //USE_CONV_POOL_DNNLOWP_OPERATOR_BASE_FUNCTIONS(std::uint8_t, ConvFp32Op);
    base: ConvPoolDNNLowPOpBase<u8, ConvFp32Op>,

    /// Save quantized weights right after quantization before layout packing for
    /// performance purpose
    save_unpacked_weights:  bool,

    quantize_groupwise:     bool,

    /// only for DNNLOWP_ACC16
    nbits_in_non_outlier:   i32,
}

input_tags!{
    ConvDNNLowPPackWeightOp
    {
        Filter,
        Bias
    }
}

register_cpu_operator_with_engine!{
    Int8ConvPackWeight,
    DNNLOWP,
    ConvDNNLowPPackWeightOp
}

///Prepack weight for Int8Conv
register_cpu_operator_with_engine!{
    Int8ConvPackWeight,
    DNNLOWP_ACC16,
    ConvDNNLowPPackWeightOp
}

impl ConvDNNLowPPackWeightOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : ConvPoolDNNLowPOpBase<uint8_t, ConvFp32Op>(operator_def, ws),
        save_unpacked_weights_(
            this->GetSingleArgument<bool>("save_unpacked_weights", false)),
            quantize_groupwise_(
                this->GetSingleArgument<bool>("quantize_groupwise", false)) 

                if (this->debug_def().engine() == "DNNLOWP_ACC16") {
                    nbits_in_non_outlier_ = this->GetSingleArgument<int>(
                        "nbits_in_non_outlier", FLAGS_caffe2_dnnlowp_nbits_in_non_outlier);
                }
        */
    }
    
    #[inline] pub fn get_external_tensor_info(&mut self, 
        c:        *const c_void,
        capacity: *mut usize,
        device:   *mut DeviceOption) -> Vec<i64> {
        
        todo!();
        /*
            const Int8ConvDNNLowPPackedWeightBlob* int8_tensor =
                reinterpret_cast<const Int8ConvDNNLowPPackedWeightBlob*>(c);
            return GetTensorInfo(&(int8_tensor->original_tensor), capacity, device);
        */
    }

    #[inline] pub fn setup_external_tensor_descriptor(&mut self, 
        blob:        *const Blob,
        shapes:      *mut Vec<Vec<u64>>,
        all_scales:  *mut Vec<Vec<f32>>,
        all_offsets: *mut Vec<Vec<i32>>,
        desc:        *mut ExternalTensorDescriptor)  {

        todo!();
        /*
            const auto& dnntensor = blob->template Get<Int8ConvDNNLowPPackedWeightBlob>();
            const Tensor& cpu_tensor = dnntensor.original_tensor;

            if (cpu_tensor.template IsType<uint8_t>()) {
                desc->dataType = kONNXIFI_DATATYPE_UINT8;
                desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<uint8_t>());
            } else if (cpu_tensor.template IsType<int32_t>()) {
                desc->dataType = kONNXIFI_DATATYPE_INT32;
                desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<int32_t>());
            } else if (cpu_tensor.template IsType<int8_t>()) {
                desc->dataType = kONNXIFI_DATATYPE_INT8;
                desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<int8_t>());
            } else {
                CAFFE_THROW(
                    "Unsupported Int8ConvDNNLowPPackedWeightBlob type in ONNXIFI: ",
                    cpu_tensor.dtype().name());
            }

            desc->quantizationParams = dnntensor.qparams.size();
            desc->quantizationAxis = 1;
            std::vector<float> scales;
            std::vector<int32_t> offsets;
            for (const auto v : dnntensor.qparams) {
                scales.push_back(v.scale);
                int32_t cur_offset = v.zero_point;
                offsets.push_back(cur_offset);
            }
            all_scales->push_back(scales);
            all_offsets->push_back(offsets);
            desc->scales = all_scales->back().data();
            desc->biases = all_offsets->back().data();

            // Set up dim and shape
            const auto shape = cpu_tensor.sizes();
            desc->dimensions = shape.size();
            shapes->emplace_back(shape.cbegin(), shape.cend());
            desc->shape = shapes->back().data();

            // not an offline tensor
            desc->isOffline = 0;
        */
    }
    
    #[cfg(use_fbgemm)]
    #[inline] pub fn get_conv_param(&mut self) -> fbgemm::conv_param_t<> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(this->kernel_.size(), 2);

        auto& filter = InputTensorCPU_(FILTER);
        const int M = filter.dim32(0), C = filter.dim32(filter.dim() - 1) * group_;

        return fbgemm::conv_param_t<>(
            1, // dummy
            C,
            M,
            {this->kernel_[0] * this->stride_[0],
            this->kernel_[1] * this->stride_[1]}, // dummy
            group_,
            {this->kernel_[0], this->kernel_[1]},
            {this->stride_[0], this->stride_[1]},
            {this->pads_[0], this->pads_[1], this->pads_[2], this->pads_[3]});
        */
    }
    
    #[cfg(use_fbgemm)]
    #[inline] pub fn get_conv3d_param(&mut self) -> fbgemm::conv_param_t<3> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);

        auto& filter = InputTensorCPU_(FILTER);
        const int M = filter.dim32(0), C = filter.dim32(filter.dim() - 1) * group_;

        return fbgemm::conv_param_t<3>(
            1, // dummy
            C,
            M,
            {1,
            this->kernel_[1] * this->stride_[1],
            this->kernel_[2] * this->stride_[2]}, // dummy
            group_,
            {this->kernel_[0], this->kernel_[1], this->kernel_[2]},
            {this->stride_[0], this->stride_[1], this->stride_[2]},
            {this->pads_[0],
            this->pads_[1],
            this->pads_[2],
            this->pads_[3],
            this->pads_[4],
            this->pads_[5]});
        */
    }
    
    #[inline] pub fn takeg_conv_fast_path(&mut self) -> bool {
        
        todo!();
        /*
            if (this->debug_def().engine() == "DNNLOWP_ACC16" ||
            (this->kernel_.size() != 2 && this->kernel_.size() != 3)) {
            return false;
        }

        if (this->kernel_.size() == 2) {
            return fbgemm::fbgemmOptimizedGConv(GetConvParam_());
        } else {
            CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
            return fbgemm::fbgemmOptimizedGConv(GetConv3DParam_());
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& filter = InputTensorCPU_(FILTER);

        auto* Y = this->Output<Int8ConvDNNLowPPackedWeightBlob>(0);
        // Create tensor with the same shape but this new tensor shouldn't actually
        // allocate memory for the tensor.
        // This is just a convenient way to pass tensor shape information
        Y->original_tensor.ResizeLike(filter);

        // Assume KRSC layout
        // The number of output channels
        int M = filter.dim32(0);
        // The number of input channels per group
        int C_per_group = filter.dim32(filter.dim() - 1);

        int kernel_dims_size = 1;
        for (int i = 0; i < filter.dim() - 2; ++i) {
            kernel_dims_size *= filter.dim32(i + 1);
        }
        int kernel_dim = C_per_group * kernel_dims_size;

        vector<int8_t> W_quantized;
        Y->qparams.resize(quantize_groupwise_ ? group_ : 1);
        QuantizeWeight<uint8_t>(
            InputBlob(FILTER),
            kernel_dim,
            M,
            Y->qparams,
            W_quantized,
            qfactory_.get());
        if (save_unpacked_weights_) {
            ReinitializeTensor(
                &Y->original_tensor, filter.sizes(), at::dtype<int8_t>().device(CPU));
            auto* buffer = Y->original_tensor.template mutable_data<int8_t>();
            CAFFE_ENFORCE_EQ(Y->original_tensor.numel(), W_quantized.size());
            memcpy(buffer, W_quantized.data(), W_quantized.size() * sizeof(int8_t));
        }

        if (this->InputIsType<int8::Int8TensorCPU>(FILTER) && quantize_groupwise_) {
            static int log_occurences = 0;
            if (log_occurences < 32) {
                ++log_occurences;
                LOG(WARNING) << "Cannot do group-wise quantization for "
                    "pre-quantized weight "
                    << this->debug_def().input(0);
            }
        }

        // Pre-compute column offsets
        // This should happen before ExtractOutlierMatrix because W_quantized is
        // changed in ExtractOutlierMatrix.
        Y->column_offsets.reset(new vector<int32_t>());
        ComputeColumnOffsets(
            kernel_dim, M, W_quantized.data(), Y->qparams, *Y->column_offsets);

        // Check if we should fallback to 32-bit accumulation.
        // This check is only meaningful when engine is DNNLOWP_ACC16.
        bool fallback_to_32_bit_accumulation = false;
        if (nbits_in_non_outlier_ == 0) {
            LOG(INFO) << "nbits_in_non_outlier == 0 means everything is outlier so we "
                "fallback to acc32";
            fallback_to_32_bit_accumulation = true;
        }
        // In Skylake, acc16 is not faster when N or K is smaller than 128
        // FIXME : code duplication with conv_dnnlowp_acc16_op.cc
        constexpr int SKYLAKE_ACC16_N_THRESHOLD_MIN = 128,
        SKYLAKE_ACC16_K_THRESHOLD_MIN = 128;
        int acc16_n_threshold = FLAGS_caffe2_dnnlowp_acc16_n_threshold;
        if (caffe2::GetCpuId().avx512f() &&
            acc16_n_threshold < SKYLAKE_ACC16_N_THRESHOLD_MIN) {
            acc16_n_threshold = SKYLAKE_ACC16_N_THRESHOLD_MIN;
        }
        int acc16_k_threshold = FLAGS_caffe2_dnnlowp_acc16_k_threshold;
        if (caffe2::GetCpuId().avx512f() &&
            acc16_k_threshold < SKYLAKE_ACC16_K_THRESHOLD_MIN) {
            acc16_k_threshold = SKYLAKE_ACC16_K_THRESHOLD_MIN;
        }
        if (!fallback_to_32_bit_accumulation && M / group_ < acc16_n_threshold) {
            LOG(INFO) << "N " << M / group_ << " of weight blob "
                << this->debug_def().input(0) << " is smaller than threshold "
                << acc16_n_threshold << " . Falling back to acc32";
            fallback_to_32_bit_accumulation = true;
        }
        if (!fallback_to_32_bit_accumulation && kernel_dim < acc16_k_threshold) {
            LOG(INFO) << "K " << kernel_dim << " of weight blob "
                << this->debug_def().input(0) << " is smaller than threshold "
                << acc16_k_threshold << " . Falling back to acc32";
            fallback_to_32_bit_accumulation = true;
        }

        // When nbits_in_non_outlier == 0, we fall back to acc32
        if (this->debug_def().engine() == "DNNLOWP_ACC16" &&
            !fallback_to_32_bit_accumulation) {
            if (nbits_in_non_outlier_ < 8) {
                int outlier_cnt = CountOutliers(
                    group_, kernel_dim, M, nbits_in_non_outlier_, W_quantized);

                LOG(INFO) << "Proportion of outlier for Conv layer with weight blob "
                    << this->debug_def().input(0) << " is "
                    << static_cast<float>(outlier_cnt) / W_quantized.size();
                LOG(INFO) << "nbits_in_non_outlier " << nbits_in_non_outlier_;

                if (static_cast<float>(outlier_cnt) / W_quantized.size() >
                    FLAGS_caffe2_dnnlowp_acc16_density_threshold) {
                    LOG(INFO) << "Density of outliers is higher than threshold "
                        << FLAGS_caffe2_dnnlowp_acc16_density_threshold
                        << " . Falling back to acc32";
                    fallback_to_32_bit_accumulation = true;
                } else {
                    Y->W_outlier.reset(ExtractOutlierMatrix(
                            group_, kernel_dim, M, nbits_in_non_outlier_, W_quantized));
                }
            }

            if (!fallback_to_32_bit_accumulation) {
                Y->nbits_in_non_outlier = nbits_in_non_outlier_;
                Y->W_acc16.reset(new fbgemm::PackBMatrix<int8_t, int16_t>(
                        fbgemm::matrix_op_t::Transpose,
                        group_ * kernel_dim,
                        M / group_,
                        W_quantized.data(),
                        kernel_dim,
                        nullptr, // pmat
                        group_));
            }
        }

        if (fallback_to_32_bit_accumulation) {
            Y->W_acc16.reset();
            Y->W_outlier.reset();
        }

        if (this->debug_def().engine() != "DNNLOWP_ACC16" ||
            fallback_to_32_bit_accumulation) {
            // acc32
            if (TakeDepthWise3x3FastPath_()) {
                Y->W_depthwise.reset(new fbgemm::PackedDepthWiseConvMatrix(
                        group_, 3 * 3, W_quantized.data()));
            } else if (TakeDepthWise3x3x3FastPath_()) {
                Y->W_depthwise.reset(new fbgemm::PackedDepthWiseConvMatrix(
                        group_, 3 * 3 * 3, W_quantized.data()));
            } else if (TakeGConvFastPath_()) {
                if (this->kernel_.size() == 2) {
                    Y->W_gconv.reset(new fbgemm::PackWeightMatrixForGConv<int8_t>(
                            fbgemm::matrix_op_t::Transpose,
                            GetConvParam_(),
                            W_quantized.data()));
                } else {
                    CAFFE_ENFORCE_EQ(this->kernel_.size(), 3);
                    Y->W_gconv3d.reset(
                        new fbgemm::PackWeightMatrixForGConv<int8_t, int32_t, 3>(
                            fbgemm::matrix_op_t::Transpose,
                            GetConv3DParam_(),
                            W_quantized.data()));
                }
            } else {
                Y->W.reset(new fbgemm::PackBMatrix<int8_t>(
                        fbgemm::matrix_op_t::Transpose,
                        group_ * kernel_dim,
                        M / group_,
                        W_quantized.data(),
                        kernel_dim,
                        nullptr, // pmat
                        group_));
            }
        }

        if (InputSize() >= 2) {
            TensorQuantizationParams in_qparams;
            CAFFE_ENFORCE(HasSingleArgumentOfType<float>("in_scale"));
            in_qparams.scale = GetSingleArgument<float>("in_scale", 0);
            Y->bias.reset(new vector<int32_t>());
            QuantizeConvBias(InputBlob(BIAS), M, in_qparams, Y->qparams, *Y->bias);
        } else {
            Y->bias = nullptr;
        }

        return true;
        */
    }
    
    #[inline] pub fn take_depth_wise3x_3fast_path(&mut self) -> bool {
        
        todo!();
        /*
            const auto& filter = this->InputTensorCPU_(FILTER);
        // The number of output channels
        int M = filter.dim32(0);
        // The number of input channels per group
        int C_per_group = filter.dim32(filter.dim() - 1);
        return this->debug_def().engine() != "DNNLOWP_ACC16" && group_ == M &&
            C_per_group == 1 && group_ % 8 == 0 && this->kernel_.size() == 2 &&
            kernel_h() == 3 && kernel_w() == 3 && stride_h() == stride_w() &&
            (stride_h() == 1 || stride_h() == 2) && dilation_h() == 1 &&
            dilation_w() == 1 && pad_t() == 1 && pad_b() == 1 && pad_l() == 1 &&
            pad_r() == 1 && GetCpuId().avx2();
        */
    }
    
    #[inline] pub fn take_depth_wise3x3x_3fast_path(&mut self) -> bool {
        
        todo!();
        /*
            const auto& filter = this->InputTensorCPU_(FILTER);
        // The number of output channels
        int M = filter.dim32(0);
        // The number of input channels per group
        int C_per_group = filter.dim32(filter.dim() - 1);
        bool ret = this->debug_def().engine() != "DNNLOWP_ACC16" && group_ == M &&
            C_per_group == 1 && group_ % 8 == 0 && this->kernel_.size() == 3 &&
            this->kernel_[0] == 3 && this->kernel_[1] == 3 && this->kernel_[2] == 3 &&
            (this->stride_[0] == 1 || this->stride_[0] == 2) &&
            (this->stride_[1] == 1 || this->stride_[1] == 2) &&
            (this->stride_[2] == 1 || this->stride_[2] == 2) &&
            this->dilation_[0] == 1 && this->dilation_[1] == 1 &&
            this->dilation_[2] == 1 &&
            accumulate(
                this->pads_.begin(), this->pads_.end(), 1, multiplies<int>()) == 1 &&
            GetCpuId().avx2();
        return ret;
        */
    }
}

/**
  | Helper functions for packing weights
  | that can be used by ConvDNNLowPAcc16PackWeightOp,
  | 
  | ConvDNNLowPOp, and ConvDNNLowPAcc16Op
  |
  */

/**
  | Set up used onnxifi data type constexpr
  | 
  | Should always be synced with onnxifi.h
  |
  */
pub const kONNXIFI_DATATYPE_UINT8: u64 = 2;
pub const kONNXIFI_DATATYPE_INT32: u64 = 6;
pub const kONNXIFI_DATATYPE_INT8: u64 = 3;

///-----------------------
pub struct Int8ConvDNNLowpPackedWeightBlobShapeFunctions { }

impl Int8ConvDNNLowpPackedWeightBlobShapeFunctions {

    pub fn new() -> Self {
    
        todo!();
        /*
            : ExternalTensorFunctionsBase()
        */
    }
}

impl ExternalTensorFunctionsBase for Int8ConvDNNLowpPackedWeightBlobShapeFunctions {

    #[inline] fn get_type_meta_id(&self) -> TypeIdentifier {
        
        todo!();
        /*
            return TypeMeta::Id<Int8ConvDNNLowPPackedWeightBlob>();
        */
    }

    #[inline] fn get_external_tensor_type(&self, c: *const c_void) -> TypeMeta {
        
        todo!();
        /*
            const Int8ConvDNNLowPPackedWeightBlob* int8_tensor =
            reinterpret_cast<const Int8ConvDNNLowPPackedWeightBlob*>(c);
        // return (int8_tensor->original_tensor).dtype();
        return TypeMeta::Make<uint8_t>();
        */
    }
    
    #[inline] fn is_quantized(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] fn is_same_meta_type(&self, id: TypeIdentifier) -> bool {
        
        todo!();
        /*
            return id == TypeMeta::Id<Int8ConvDNNLowPPackedWeightBlob>();
        */
    }
    

    #[inline] fn load_info_of_blob(&self, 
        blob:   *const Blob,
        scale:  *mut Vec<f32>,
        offset: *mut Vec<f32>,
        axis:   *mut u32)  {

        todo!();
        /*
            scale->clear();
        offset->clear();
        const Int8ConvDNNLowPPackedWeightBlob* int8_tensor =
            reinterpret_cast<const Int8ConvDNNLowPPackedWeightBlob*>(blob->GetRaw());
        const auto& qparams = int8_tensor->qparams;
        for (const auto& qparam : qparams) {
            scale->emplace_back(qparam.scale);
            offset->emplace_back(static_cast<float>(qparam.zero_point));
        }
        *axis = 1;
        */
    }
}

///-----------------------
pub struct Int8FCDNNLowpPackedWeightBlobShapeFunctions { }

impl Int8FCDNNLowpPackedWeightBlobShapeFunctions {

    pub fn new() -> Self {
    
        todo!();
        /*
            : ExternalTensorFunctionsBase()
        */
    }
}

impl ExternalTensorFunctionsBase for Int8FCDNNLowpPackedWeightBlobShapeFunctions {

    #[inline] fn is_quantized(&self) -> bool {

        todo!();
        /*
           return true;
           */
    }

     #[inline] fn get_external_tensor_info(&mut self, 
        c:        *const c_void,
        capacity: *mut usize,
        device:   *mut DeviceOption) -> Vec<i64> {

        todo!();
        /*
            const Int8FCDNNLowPPackedWeightBlob* int8_tensor =
                reinterpret_cast<const Int8FCDNNLowPPackedWeightBlob*>(c);
            return GetTensorInfo(&(int8_tensor->original_tensor), capacity, device);
        */
    }
    
    
    #[inline] fn load_info_of_blob(&self, 
        blob:   *const Blob,
        scale:  *mut Vec<f32>,
        offset: *mut Vec<f32>,
        axis:   *mut u32)  {

        todo!();
        /*
            scale->clear();
        offset->clear();
        const Int8FCDNNLowPPackedWeightBlob* int8_tensor =
            reinterpret_cast<const Int8FCDNNLowPPackedWeightBlob*>(blob->GetRaw());
        const auto& qparams = int8_tensor->qparams;
        for (const auto& qparam : qparams) {
            scale->emplace_back(qparam.scale);
            offset->emplace_back(static_cast<float>(qparam.zero_point));
        }
        *axis = 1;
        */
    }
    
    #[inline] fn is_same_meta_type(&self, id: TypeIdentifier) -> bool {
        
        todo!();
        /*
            return id == TypeMeta::Id<Int8FCDNNLowPPackedWeightBlob>();
        */
    }
    
    #[inline] fn get_type_meta_id(&self) -> TypeIdentifier {
        
        todo!();
        /*
            return TypeMeta::Id<Int8FCDNNLowPPackedWeightBlob>();
        */
    }
    
    #[inline] fn get_external_tensor_type(&self, c: *const c_void) -> TypeMeta {
        
        todo!();
        /*
            const Int8FCDNNLowPPackedWeightBlob* int8_tensor =
            reinterpret_cast<const Int8FCDNNLowPPackedWeightBlob*>(c);
        // We forced the output type to be uint8_t since we know it always is.
        // If it is going to be implemented elsewhere, we might need to change here.
        // return (int8_tensor->original_tensor).dtype();
        return TypeMeta::Make<uint8_t>();
        */
    }
    
    #[inline] fn setup_external_tensor_descriptor(&self, 
        blob:        *const Blob,
        shapes:      *mut Vec<Vec<u64>>,
        all_scales:  *mut Vec<Vec<f32>>,
        all_offsets: *mut Vec<Vec<i32>>,
        desc:        *mut ExternalTensorDescriptor)  {

        todo!();
        /*
            const auto& dnntensor = blob->template Get<Int8FCDNNLowPPackedWeightBlob>();
        const Tensor& cpu_tensor = dnntensor.original_tensor;

        if (cpu_tensor.template IsType<uint8_t>()) {
            desc->dataType = kONNXIFI_DATATYPE_UINT8;
            desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<uint8_t>());
        } else if (cpu_tensor.template IsType<int32_t>()) {
            desc->dataType = kONNXIFI_DATATYPE_INT32;
            desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<int32_t>());
        } else if (cpu_tensor.template IsType<int8_t>()) {
            desc->dataType = kONNXIFI_DATATYPE_INT8;
            desc->buffer = reinterpret_cast<uint64_t>(cpu_tensor.data<int8_t>());
        } else {
            CAFFE_THROW(
                "Unsupported Int8FCDNNLowPPackedWeightBlob type in ONNXIFI: ",
                cpu_tensor.dtype().name());
        }

        desc->quantizationParams = dnntensor.qparams.size();
        desc->quantizationAxis = 1;
        std::vector<float> scales;
        std::vector<int32_t> offsets;
        for (const auto v : dnntensor.qparams) {
            scales.push_back(v.scale);
            int32_t cur_offset = v.zero_point;
            offsets.push_back(cur_offset);
        }
        all_scales->push_back(scales);
        all_offsets->push_back(offsets);
        desc->scales = all_scales->back().data();
        desc->biases = all_offsets->back().data();

        // Set up dim and shape
        const auto shape = cpu_tensor.sizes();
        desc->dimensions = shape.size();
        shapes->emplace_back(shape.cbegin(), shape.cend());
        desc->shape = shapes->back().data();

        // not an offline tensor
        desc->isOffline = 0;
        */
    }
}

declare_int32!{caffe2_dnnlowp_nbits_in_non_outlier}
declare_double!{caffe2_dnnlowp_acc16_density_threshold}
declare_int32!{caffe2_dnnlowp_acc16_n_threshold}
declare_int32!{caffe2_dnnlowp_acc16_k_threshold}

//u8, u16
#[inline] pub fn quantize_weight<T,SIGNED_T>(
    blob:        &Blob,
    kernel_dim:  i32,
    m:           i32,
    qparams:     &mut Vec<TensorQuantizationParams>,
    w_quantized: &mut Vec<SIGNED_T>,
    qfactory:    *mut QuantizationFactory)  {

    todo!();
    /*
        using T_signed = typename make_signed<T>::type;

            const auto& filter = blob.IsType<int8::Int8TensorCPU>()
                ? blob.Get<int8::Int8TensorCPU>().t
                : blob.Get<TensorCPU>();

            W_quantized.resize(filter.numel());

            int signed_min = -(1 << (qfactory->GetWeightPrecision() - 1));
            if (blob.IsType<int8::Int8TensorCPU>()) {
                qparams[0].scale = blob.Get<int8::Int8TensorCPU>().scale;
                qparams[0].zero_point =
                    blob.Get<int8::Int8TensorCPU>().zero_point + signed_min;

                const T* W_data = filter.data<T>();
                for (auto i = 0; i < filter.numel(); ++i) {
                    W_quantized[i] = W_data[i] + signed_min;
                }
            } else {
                for (int g = 0; g < qparams.size(); ++g) {
                    size_t offset = g * (M / qparams.size()) * kernel_dim;
                    qparams[g] = qfactory->ChooseQuantizationParams(
                        filter.data<float>() + offset,
                        (M / qparams.size()) * kernel_dim,
                        true /*weight*/);

                    // qparams[g] is computed for unsigned type.
                    // Adjust for the fact that weight will actually use signed.
                    qparams[g].zero_point += signed_min;

                    fbgemm::Quantize<T_signed>(
                        filter.data<float>() + offset,
                        W_quantized.data() + offset,
                        (M / qparams.size()) * kernel_dim,
                        qparams[g]);
                }
            }
    */
}

/**
  | TODO reuse col_offsets_with_zero_pt_s8acc32_ref
  | in fbgemm RefImplementations.cc . We can't do
  | this now because W_quantized is not transposed
  | here.
  |
  | i8, i16
  */
#[inline] pub fn compute_column_offsets<T>(
    num_rows:    i32,
    num_cols:    i32,
    w:           *const T,
    qparams:     &Vec<TensorQuantizationParams>,
    col_offsets: &mut Vec<i32>)  {

    todo!();
    /*
        col_offsets.resize(num_cols);
            int num_quant_groups = qparams.size();
            for (int g = 0; g < num_quant_groups; ++g) {
                int j_begin = g * (num_cols / num_quant_groups);
                int j_end = j_begin + (num_cols / num_quant_groups);
                for (int j = j_begin; j < j_end; ++j) {
                    int32_t sum = 0;
                    for (int k = 0; k < num_rows; ++k) {
                        sum += W[j * num_rows + k];
                    }
                    col_offsets[j] = sum - qparams[g].zero_point * num_rows;
                }
            }
    */
}

#[inline] pub fn count_outliers(
    groups:               i32,
    kernel_dim:           i32,
    m:                    i32,
    nbits_in_non_outlier: i32,
    w_quantized:          &mut Vec<i8>) -> i32 {
    
    todo!();
    /*
        int outlier_cnt = 0;
        for (int group_id = 0; group_id < groups; ++group_id) {
            for (int i = 0; i < (M / groups) * kernel_dim; ++i) {
                int8_t w = W_quantized[group_id * (M / groups) * kernel_dim + i];
                bool is_outlier = nbits_in_non_outlier == 0 ||
                    w < -(1 << (nbits_in_non_outlier - 1)) ||
                    w >= (1 << (nbits_in_non_outlier - 1));
                if (is_outlier) {
                    ++outlier_cnt;
                }
            }
        }
        return outlier_cnt;
    */
}

/**
  | -----------
  | @param W_quantized
  | 
  | input quantized weight that is not packed
  | yet
  |
  */
#[cfg(use_fbgemm)]
#[inline] pub fn extract_outlier_matrix(
    groups:               i32,
    kernel_dim:           i32,
    m:                    i32,
    nbits_in_non_outlier: i32,
    w_quantized:          &mut Vec<i8>) -> *mut fbgemm::CompressedSparseColumn {
    
    todo!();
    /*
        int outlier_cnt =
            CountOutliers(groups, kernel_dim, M, nbits_in_non_outlier, W_quantized);

        fbgemm::CompressedSparseColumn* Wq_outlier =
            new fbgemm::CompressedSparseColumn(kernel_dim, M);
        Wq_outlier->RowIdx().resize(outlier_cnt);
        Wq_outlier->Values().resize(outlier_cnt);

        outlier_cnt = 0;
        for (int group_id = 0; group_id < groups; ++group_id) {
            for (int j = 0; j < M / groups; ++j) {
                Wq_outlier->ColPtr()[group_id * (M / groups) + j] = outlier_cnt;

                for (int k = 0; k < kernel_dim; ++k) {
                    int8_t w = W_quantized[(group_id * (M / groups) + j) * kernel_dim + k];
                    bool is_outlier = nbits_in_non_outlier == 0 ||
                        w < -(1 << (nbits_in_non_outlier - 1)) ||
                        w >= (1 << (nbits_in_non_outlier - 1));
                    if (is_outlier) {
                        CAFFE_ENFORCE_LE(k, int16_t::max);
                        Wq_outlier->RowIdx()[outlier_cnt] = k;
                        Wq_outlier->Values()[outlier_cnt] = w;
                        ++outlier_cnt;

                        W_quantized[(group_id * (M / groups) + j) * kernel_dim + k] = 0;
                    }
                }
            }
        } // for each group
        CAFFE_ENFORCE_EQ(outlier_cnt, Wq_outlier->RowIdx().size());
        Wq_outlier->ColPtr()[M] = outlier_cnt;

        return Wq_outlier;
    */
}


// FIXME: code duplication with ConvDNNLowPOp::QuantizeBias_
#[inline] pub fn quantize_conv_bias(
    blob:                  &Blob,
    m:                     i32,
    in_qparams:            &TensorQuantizationParams,
    filter_qparams:        &Vec<TensorQuantizationParams>,
    b_quantized:           &mut Vec<i32>,
    use_fp16:              Option<bool>,
    round_to_nearest_even: Option<bool>)  {

    let use_fp16:           bool = use_fp16.unwrap_or(false);
    let round_to_nearest_even: bool = round_to_nearest_even.unwrap_or(true);
    
    todo!();
    /*
        const auto& bias = blob.IsType<int8::Int8TensorCPU>()
            ? blob.Get<int8::Int8TensorCPU>().t
            : blob.Get<TensorCPU>();
        if (blob.IsType<int8::Int8TensorCPU>()) {
            TensorQuantizationParams bias_qparams;
            bias_qparams.scale = blob.Get<int8::Int8TensorCPU>().scale;
            bias_qparams.zero_point = blob.Get<int8::Int8TensorCPU>().zero_point;
            CAFFE_ENFORCE_LE(
                std::abs(
                    bias_qparams.scale - in_qparams.scale * filter_qparams[0].scale),
                    1e-4);
            CAFFE_ENFORCE_EQ(bias_qparams.zero_point, 0);
            b_quantized.resize(bias.numel());
            b_quantized.assign(
                bias.data<int32_t>(), bias.data<int32_t>() + bias.numel());
        } else {
            const float* bdata = bias.data<float>();
            vector<float> bdata_local;
            if (use_fp16) {
                bdata_local.resize(bias.numel());
                fbgemm::RoundToFloat16(
                    bdata, bdata_local.data(), bias.numel(), false /* FLAGS_caffe2_fbgemm_fake_fp16_clamp */);
                bdata = bdata_local.data();
            }
            b_quantized.resize(bias.numel());
            for (int g = 0; g < filter_qparams.size(); ++g) {
                int i_begin = g * (M / filter_qparams.size());
                int i_end = i_begin + (M / filter_qparams.size());
                for (int i = i_begin; i < i_end; ++i) {
                    if (round_to_nearest_even) {
                        b_quantized[i] = fbgemm::Quantize<int32_t>(
                            bdata[i],
                            0,
                            in_qparams.scale * filter_qparams[g].scale,
                            32,
                            true /* signed */);
                    } else {
                        b_quantized[i] = round((1.0f / in_qparams.scale) * (1.0f / filter_qparams[g].scale) * bdata[i]);
                        b_quantized[i] = std::max(std::min(b_quantized[i], INT32_MAX), INT32_MIN);
                    }
                }
            }
        }
    */
}

// Explicitly register TypeMeta
caffe_known_type!{Int8FCDNNLowPPackedWeightBlob}
caffe_known_type!{Int8ConvDNNLowPPackedWeightBlob}

// Register DNNLOWP Type in caffe2 core
register_external_tensor_functions!{
    TypeMeta::Id::<Int8FCDNNLowPPackedWeightBlob>(),
    Int8FCDNNLowpPackedWeightBlobShapeFunctions
}

register_external_tensor_functions!{
    TypeMeta::Id::<Int8ConvDNNLowPPackedWeightBlob>(),
    Int8ConvDNNLowpPackedWeightBlobShapeFunctions
}

///---------------------
///Prepack weight for Int8FC
register_cpu_operator!{
    Int8FCPackWeight, 
    FullyConnectedDNNLowPPackWeightOp
}

register_cpu_operator_with_engine!{
    Int8FCPackWeight,
    DNNLOWP,
    FullyConnectedDNNLowPPackWeightOp
}

register_cpu_operator_with_engine!{
    Int8FCPackWeight,
    DNNLOWP_ACC16,
    FullyConnectedDNNLowPPackWeightOp
}

register_cpu_operator_with_engine!{
    Int8FCPackWeight,
    DNNLOWP_ROWWISE,
    FullyConnectedDNNLowPPackWeightOp
}

num_inputs!{Int8FCPackWeight, (1,2)}

num_outputs!{Int8FCPackWeight, (1,2)}

inputs!{Int8FCPackWeight, 
    0 => ("W",                        "Weight tensor in KRSC layout"),
    1 => ("b",                        "Bias tensor")
}

outputs!{Int8FCPackWeight, 
    0 => ("W_q",                      "Weight/bias tensor in a packed format with type Int8FCDNNLowPPackedWeightBlob"),
    1 => ("B_q",                      "Bias int32 quantized tensor")
}

args!{Int8FCPackWeight, 
    0 => ("axis_w",                   "See FC operator"),
    1 => ("quantize_channelwise",     "Default false. Per output channel quantization"),
    2 => ("save_unpacked_weights",    "Default false. Store unpacked quantized weights to W_q.original_tensor"),
    3 => ("in_scale",                 "The scale of input activation tensor. Only meaningful when bias is provided (NOTE: this is not the scale of weight")
}

tensor_inference_function!{Int8FCPackWeight, /* ([](const OperatorDef& def,
            const vector<TensorShape>& in) {
        vector<TensorShape> out;
        TensorShape W = in[0];
        out.emplace_back(std::move(W));
        out[0].set_data_type(TensorProto_DataType_INT8);
        if (def.output_size() > 1) {
            TensorShape b = in[1];
            out.emplace_back(std::move(b));
            out[1].set_data_type(TensorProto_DataType_INT32);
        }
        return out;
    }) */}

num_inputs!{Int8ConvPackWeight, (1,2)}

num_outputs!{Int8ConvPackWeight, 1}

inputs!{Int8ConvPackWeight, 
    0 => ("W", "Weight tensor in KRSC layout"),
    1 => ("b", "Bias tensor")
}

outputs!{Int8ConvPackWeight, 
    0 => ("W_q", "Weight/bias tensor in a packed format with type Int8ConvDNNLowPPackedWeightBlob")
}

args!{Int8ConvPackWeight, 
    0 => ("quantize_groupwise",    "Default false. Per group quantization"),
    1 => ("save_unpacked_weights", "Default false. Store unpacked quantized weights to W_q.original_tensor"),
    2 => ("in_scale",              "The scale of input activation tensor. Only meaningful when bias is provided (NOTE: this is not the scale of weight")
}

tensor_inference_function!{Int8ConvPackWeight, /* ([](const OperatorDef& def,
            const vector<TensorShape>& in) {
        vector<TensorShape> out;
        TensorShape W = in[0];
        out.emplace_back(std::move(W));
        out[0].set_data_type(TensorProto_DataType_INT8);
        if (def.output_size() > 1) {
            TensorShape b = in[1];
            out.emplace_back(std::move(b));
            out[1].set_data_type(TensorProto_DataType_INT32);
        }
        return out;
    }) */
}
