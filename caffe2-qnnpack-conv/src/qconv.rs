crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qconv.cpp]

/**
  | To have a sanity check for maximum matrix
  | size.
  |
  */
pub const K_REASONABLE_MAX_DIM: i64 = 1000000;

pub fn conv_dim_checks<const SPATIAL_DIM: usize = 2>(
        act_dims:            i64,
        stride_dims:         i64,
        padding_dims:        i64,
        output_padding_dims: i64,
        dilation_dims:       i64,
        func_name:           String,
        transpose:           bool) -> bool {
    let transpose: bool = transpose.unwrap_or(false);

    todo!();
        /*
            TORCH_CHECK(
          act_dims == kSpatialDim + 2,
          func_name,
          kSpatialDim,
          "d(): Expected activation tensor to have ",
          kSpatialDim + 2,
          " dimensions, got ",
          act_dims);
      TORCH_CHECK(
          stride_dims == kSpatialDim,
          func_name,
          kSpatialDim,
          "d(): Expected stride tensor to have ",
          kSpatialDim,
          " dimensions, got ",
          stride_dims);
      TORCH_CHECK(
          padding_dims == kSpatialDim,
          func_name,
          kSpatialDim,
          "d(): Expected padding tensor to have ",
          kSpatialDim,
          " dimensions, got ",
          padding_dims);
      TORCH_CHECK(
          !transpose || (output_padding_dims == kSpatialDim),
          func_name,
          kSpatialDim,
          "d(): Expected output padding tensor to have ",
          kSpatialDim,
          " dimensions, got ",
          output_padding_dims);
      TORCH_CHECK(
          dilation_dims == kSpatialDim,
          func_name,
          kSpatialDim,
          "d(): Expected dilation tensor to have ",
          kSpatialDim,
          " dimensions, got ",
          dilation_dims);
      return true;
        */
}

#[inline] pub fn compute_deconv_shape(
        input:          i64,
        kernel:         i64,
        stride:         i64,
        input_padding:  i64,
        output_padding: i64,
        dilation:       i64) -> i64 {
    
    todo!();
        /*
            i64 out = (input - 1) * stride - 2 * input_padding
                    + dilation * (kernel - 1) + output_padding + 1;
      return out;
        */
}

pub fn make_de_conv_output_shape<const kSpatialDim: i64>(
        N:              i64,
        M:              i64,
        input_shape:    &Vec<i64>,
        kernel:         &Vec<i64>,
        stride:         &TorchList<i64>,
        input_padding:  &TorchList<i64>,
        output_padding: &TorchList<i64>,
        dilation:       &TorchList<i64>) -> SmallVector<i64,{SPATIAL_DIM + 2}> {

    todo!();
        /*
            SmallVector<i64, kSpatialDim + 2> output_shape;
      output_shape.resize(kSpatialDim + 2);
      output_shape[0] = N;  // Batch size
      output_shape[1] = M;  // Output channels
      for (i64 idx = 0; idx < kSpatialDim; ++idx) {
        output_shape[idx + 2] = compute_deconv_shape(input_shape[idx],
                                                     kernel[idx],
                                                     stride[idx],
                                                     input_padding[idx],
                                                     output_padding[idx],
                                                     dilation[idx]);
        TORCH_CHECK(output_shape[idx + 2] > 0,
                    "Output dimension is zero for ", idx, " axis;"
                    " kernel: ", kernel[idx],
                    ", stride: ", stride[idx],
                    ", input padding: ", input_padding[idx],
                    ", output padding: ", output_padding[idx],
                    ", dilation: ", dilation[idx])
        TORCH_CHECK(output_shape[idx + 2] < kReasonableMaxDim,
                    "Output dimension is beyound reasonable maximum for ", idx,
                    " axis;"
                    " kernel: ", kernel[idx],
                    ", stride: ", stride[idx],
                    ", input padding: ", input_padding[idx],
                    ", output padding: ", output_padding[idx],
                    ", dilation: ", dilation[idx]);
      }
      return output_shape;
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_conv_output_shape<const SPATIAL_DIM: usize = 2>(
    N:                  i32,
    M:                  i32,
    output_image_shape: &[i32; SPATIAL_DIM]) -> SmallVector<i64,{SPATIAL_DIM + 2}> {

    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_conv_output_shape2(
    N:                  i32,
    M:                  i32,
    output_image_shape: &[i32; 2]) -> SmallVector<i64,4> {
    
    todo!();
        /*
            return {N, M, output_image_shape[0], output_image_shape[1]};
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_conv_output_shape3(
        N:                  i32,
        M:                  i32,
        output_image_shape: &[i32; 3]) -> SmallVector<i64,5> {
    
    todo!();
        /*
            return {N,
              M,
              output_image_shape[0],
              output_image_shape[1],
              output_image_shape[2]};
        */
}

#[cfg(feature = "fbgemm")]
#[cfg(USE_PYTORCH_QNNPACK)]
pub fn make_conv_output_shape<const SPATIAL_DIM: usize>(

    /// mini-batch
    N:                 i32,

    /// output channels
    M:                 i32,
    input_image_shape: &Vec<i32>,
    kernel:            &Vec<i64>,
    stride:            &TorchList<i64>,
    padding:           &TorchList<i64>,
    dilation:          &TorchList<i64>) -> SmallVector<i64,SpatialDimPlusTwo> {

    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
#[cfg(USE_PYTORCH_QNNPACK)]
pub fn make_conv_output_shape2(

    /// mini-batch
    N:                 i32,

    /// output channels
    M:                 i32,

    input_image_shape: &Vec<i32>,
    kernel:            &Vec<i64>,
    stride:            &TorchList<i64>,
    padding:           &TorchList<i64>,
    dilation:          &TorchList<i64>) -> SmallVector<i64,4> {

    todo!();
        /*
            const int H = input_image_shape[0];
      const int W = input_image_shape[1];
      const i64 Y_H =
          (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
      const i64 Y_W =
          (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
      return {N, M, Y_H, Y_W};
        */
}

#[cfg(feature = "fbgemm")]
#[cfg(USE_PYTORCH_QNNPACK)]
pub fn make_conv_output_shape3(

    /// mini-batch
    N:                 i32,

    /// output channels
    M:                 i32,

    input_image_shape: &Vec<i32>,
    kernel:            &Vec<i64>,
    stride:            &TorchList<i64>,
    padding:           &TorchList<i64>,
    dilation:          &TorchList<i64>) -> SmallVector<i64,5> {

    todo!();
        /*
            const int D = input_image_shape[0];
      const int H = input_image_shape[1];
      const int W = input_image_shape[2];
      const i64 Y_D =
          (D + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
      const i64 Y_H =
          (H + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
      const i64 Y_W =
          (W + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1;
      return {N, M, Y_D, Y_H, Y_W};
        */
}

#[cfg(feature = "fbgemm")]
impl<const kSpatialDim: i32> PackedConvWeight<kSpatialDim> {
    
    pub fn get_bias_data(&mut self, bias_ptr: *mut Tensor) -> *const f32 {
    
        todo!();
        /*
            const float* bias_data = nullptr;
      if (bias.has_value()) {
        *bias_ptr = bias.value();
        TORCH_CHECK(
            bias_ptr->dtype() == kFloat,
            "[QConv3D] The 'bias' tensor must have 'torch.float' dtype");
        *bias_ptr = bias_ptr->contiguous();
        TORCH_CHECK(bias_ptr->dim() == 1, "bias should be a vector (1D Tensor)");
        const int M = w->outputChannels();
        TORCH_CHECK(bias_ptr->size(0) == M, "bias should have ", M, " elements.");
        bias_data = bias_ptr->data_ptr<float>();
      }
      return bias_data;
        */
    }
    
    pub fn get_quantization_params(&mut self, 
        act_scale:               f32,
        out_scale:               f32,
        output_multiplier_float: *mut Vec<f32>,
        act_times_w_scale:       *mut Vec<f32>)  {
        
        todo!();
        /*
            if (q_scheme == kPerTensorAffine) {
        *act_times_w_scale = {(act_scale * w_scale[0])};
        *output_multiplier_float = {act_times_w_scale->front() / out_scale};
      } else if (q_scheme == kPerChannelAffine) {
        const int M = w->outputChannels();
        output_multiplier_float->resize(M);
        act_times_w_scale->resize(M);
        for (int i = 0; i < M; ++i) {
          act_times_w_scale->at(i) = (act_scale * w_scale[i]);
          output_multiplier_float->at(i) = act_times_w_scale->at(i) / out_scale;
        }
      } else {
        TORCH_CHECK(false, "[QConv", kSpatialDim, "D] Unknown quantization scheme");
      }
        */
    }
    
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<false>(input, output_scale, output_zero_point);
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<true>(input, output_scale, output_zero_point);
        */
    }
    
    pub fn apply_impl<const RELU_FUSED: bool>(&mut self, 
        act:               &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            // Quantized kernels are all written with NHWC (channels last) layout in
      // mind. Ideally, we'd be compatible with conv2d behavior and preserve the
      // inputs layout as is (doing necessary upconversions).
      //
      // However, to be more robust, for now we just force output layout to always
      // be NHWC (channels last), thus opportunistically improving perf.
      //
      // This might change when full memory format support lands
      // See https://github.com/pytorch/pytorch/issues/23403
      const string func_name = transpose() ? "quantized::conv_transpose"
                                                : "quantized::conv";
      TORCH_CHECK(
          fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
      ConvDimChecks<kSpatialDim>(
          act.ndimension(), stride().size(), padding().size(),
          output_padding().size(), dilation().size(), func_name, transpose());

      const int N = act.size(0);
      const int C = act.size(1);
      const int D = kSpatialDim == 2 ? 1 : act.size(2);
      const int H = act.size(kSpatialDim);
      const int W = act.size(kSpatialDim + 1);

      const Tensor act_nhwc = kSpatialDim == 2
          ? act.contiguous(MemoryFormat::ChannelsLast)
          : native::fbgemm_utils::ConvertToChannelsLast3dTensor(act);
      const u8* act_data =
          reinterpret_cast<u8*>(act_nhwc.data_ptr<quint8>());
      auto* pack_w = w.get();

      const int M = pack_w->outputChannels();
      const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
      const int kernel_h = kernel[kSpatialDim - 2];
      const int kernel_w = kernel[kSpatialDim - 1];
      const int pad_d = kSpatialDim == 2 ? 0 : padding_[0];
      const int pad_h = padding_[kSpatialDim - 2];
      const int pad_w = padding_[kSpatialDim - 1];
      const int stride_d = kSpatialDim == 2 ? 1 : stride_[0];
      const int stride_h = stride_[kSpatialDim - 2];
      const int stride_w = stride_[kSpatialDim - 1];
      const int dilation_d = kSpatialDim == 2 ? 1 : dilation_[0];
      const int dilation_h = dilation_[kSpatialDim - 2];
      const int dilation_w = dilation_[kSpatialDim - 1];
      const int output_padding_d = kSpatialDim == 2 ? 0 : output_padding_[0];
      const int output_padding_h = output_padding_[kSpatialDim - 2];
      const int output_padding_w = output_padding_[kSpatialDim - 1];

      if (kSpatialDim == 2) {
        TORCH_CHECK(
            C == pack_w->inputChannels(),
            "[QConv2D] Given groups=",
            groups_,
            ", weight of size ",
            M,
            ", ",
            kernel_h,
            ", ",
            kernel_w,
            ", ",
            pack_w->inputChannels(),
            ", expected input (NCHW) ",
            N,
            ", ",
            C,
            ", ",
            H,
            ", ",
            W,
            " to have ",
            pack_w->inputChannels(),
            " channels, but got ",
            C,
            " channels instead");
      } else {
        TORCH_CHECK(
            C == pack_w->inputChannels(),
            "[QConv3D] Given groups=",
            groups_,
            ", weight of size ",
            M,
            ", ",
            kernel_d,
            ", ",
            kernel_h,
            ", ",
            kernel_w,
            ", ",
            pack_w->inputChannels(),
            ", expected input (NCDHW) ",
            N,
            ", ",
            C,
            ", ",
            D,
            ", ",
            H,
            ", ",
            W,
            " to have ",
            pack_w->inputChannels(),
            " channels, but got ",
            C,
            " channels instead");
      }

      fbgemm::conv_param_t<kSpatialDim> conv_p =
          native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
              N, // Batch size
              C, // Number of input channels
              M, // Number of output channels
              kSpatialDim == 2 ? vector<int>{H, W} : vector<int>{D, H, W},
              groups_,
              kSpatialDim == 2 ? vector<int>{kernel_h, kernel_w}
                               : vector<int>{kernel_d, kernel_h, kernel_w},
              kSpatialDim == 2 ? vector<int>{stride_h, stride_w}
                               : vector<int>{stride_d, stride_h, stride_w},
              kSpatialDim == 2 ? vector<int>{pad_h, pad_w}
                               : vector<int>{pad_d, pad_h, pad_w},
              kSpatialDim == 2
                  ? vector<int>{dilation_h, dilation_w}
                  : vector<int>{dilation_d, dilation_h, dilation_w},
              kSpatialDim == 2
                  ? vector<int>{output_padding_h, output_padding_w}
                  : vector<int>{output_padding_d,
                                     output_padding_h,
                                     output_padding_w},
              transpose());

      const float act_scale = act.q_scale();
      const i32 act_zero_point = act.q_zero_point();

      Tensor bias;
      const float* bias_data = GetBiasData(&bias);

      TORCH_CHECK(
          w_scale.size() == w_zp.size(),
          "Weight scales and zero points vectors should have the same size.");
      vector<float> output_multiplier_float;
      vector<float> act_times_w_scale;
      GetQuantizationParams(
          act_scale, output_scale, &output_multiplier_float, &act_times_w_scale);

      SmallVector<i64, kSpatialDim + 2> output_shape;
      if (transpose()) {
        output_shape = MakeDeConvOutputShape<kSpatialDim>(
            N,
            M,
            kSpatialDim == 2 ? vector<i64>{H, W} : vector<i64>{D, H, W},
            kernel,
            stride(),
            padding(),
            output_padding(),
            dilation());
      } else {
        output_shape = MakeConvOutputShape<kSpatialDim>(N, M, conv_p.OUT_DIM);
      }
      if (N > 0) {
        TORCH_CHECK(
            all_of(
                output_shape.begin(),
                output_shape.end(),
                [](i64 i) { return i > 0; }),
            "[QConv",
            kSpatialDim,
            "D] each dimension of output tensor should be greater than 0");
      }
      Tensor output = kSpatialDim == 2
          ? _empty_affine_quantized(
                output_shape,
                device(kCPU)
                    .dtype(kQUInt8)
                    .memory_format(MemoryFormat::ChannelsLast),
                output_scale,
                output_zero_point,
                nullopt)
          : native::fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
                output_shape[0],
                output_shape[1],
                output_shape[2],
                output_shape[3],
                output_shape[4],
                device(kCPU).dtype(kQUInt8),
                output_scale,
                output_zero_point);
      Tensor buffer =
          empty(output.sizes(), output.options().dtype(kInt));
      const int num_tasks = get_num_threads();
      parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
        fbgemm::DoNothing<> kNoOpObj{};
        for (const auto task_id : irange(begin, end)) {
          if (q_scheme == kPerTensorAffine) {
            fbgemm::ReQuantizeOutput<
                kReluFused,
                fbgemm::QuantizationGranularity::TENSOR,
                float>
                output_proc_obj(
                    kNoOpObj,
                    output_multiplier_float.data(),
                    output_zero_point,
                    act_zero_point,
                    w_zp.data(),
                    nullptr, /* row offset buffer */
                    col_offsets.data(),
                    bias_data,
                    M,
                    groups_,
                    act_times_w_scale.data());
            fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, i32>(
                conv_p,
                act_data,
                *pack_w,
                reinterpret_cast<u8*>(output.data_ptr<quint8>()),
                buffer.data_ptr<i32>(),
                output_proc_obj,
                task_id /* thread_id*/,
                num_tasks /* num_threads */);
          } else if (q_scheme == kPerChannelAffine) {
            fbgemm::ReQuantizeOutput<
                kReluFused,
                fbgemm::QuantizationGranularity::OUT_CHANNEL,
                float>
                output_proc_obj(
                    kNoOpObj,
                    output_multiplier_float.data(),
                    output_zero_point,
                    act_zero_point,
                    w_zp.data(),
                    nullptr, /* row offset buffer */
                    col_offsets.data(),
                    bias_data,
                    M,
                    groups_,
                    act_times_w_scale.data());

            fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, i32>(
                conv_p,
                act_data,
                *pack_w,
                reinterpret_cast<u8*>(output.data_ptr<quint8>()),
                buffer.data_ptr<i32>(),
                output_proc_obj,
                task_id /* thread_id*/,
                num_tasks /* num_threads */);
          }
        }
      });

      return output;
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl<const SPATIAL_DIM: usize> PackedConvWeightsQnnp<SPATIAL_DIM> {
    
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<false>(input, output_scale, output_zero_point);
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<true>(input, output_scale, output_zero_point);
        */
    }
    
    pub fn apply_impl<const kReluFused: bool>(&mut self, 
        act:               &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
            const string func_name = transpose() ? "quantized::conv_transpose"
                                                : "quantized::conv";
      TORCH_CHECK(!(kReluFused && transpose()),
                  kSpatialDim == 2,
                  func_name, kSpatialDim,
                  "d (qnnpack): ConvTranspose cannot be fused with ReLU.");
      TORCH_CHECK(
          kSpatialDim == 2,
          func_name, kSpatialDim,
          "d (qnnpack): QNNPACK only supports Conv2d now.");
      ConvDimChecks<kSpatialDim>(
          act.ndimension(), stride().size(), padding().size(),
          output_padding().size(), dilation().size(), func_name, transpose());

      auto* pack_w = w.get();

      // TODO Can be replaced with packB->getOutputChannels() when update pre-pack
      // to actually do the packing.
      const int out_ch_idx = transpose() ? 1 : 0;
      const auto out_ch = bias.size(0);
      // inputs are in semantic NCHW format
      const int N = act.size(0);
      const int C = act.size(1);
      const int H = act.size(2);
      const int W = act.size(3);
      const int M = out_ch; // output channels

      const Tensor act_nhwc = act.contiguous(MemoryFormat::ChannelsLast);

      auto output_min = kReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .first
          : u8::min;
      auto output_max = kReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .second
          : u8::max;

      double act_input_scale = act_nhwc.q_scale();

      // Re-quantizing the bias based on input scale and weight scale.
      if (!input_scale.has_value() || input_scale.value() != act_input_scale) {
        TORCH_CHECK(M == (transpose() ? groups() : 1) * orig_weight.size(out_ch_idx),
            "Output channel size of weight and bias must match.");
        TORCH_CHECK(C == (transpose() ? 1 : groups()) * orig_weight.size(1 - out_ch_idx),
            "Input channel size of weight and bias must match.");

        // Get the original weight and adjust it to uint8 from int8
        auto weight_contig =
            orig_weight.contiguous(MemoryFormat::ChannelsLast);
        auto bias_fp32 = bias;
        i8* w_data =
            reinterpret_cast<i8*>(weight_contig.template data_ptr<qint8>());

        float* weight_scales_data = w_scales.data_ptr<float>();
        // We calculate requant scale here as the vector holding the requant scale
        // is owned by this module. The pointer is then passed to qnnpack backend.
        generate_requantization_scales(
            w_scales, act_input_scale, output_scale, requantization_scales);

        // TODO Kimish, we are allocating affine_quantized regardless of per channel or not.
        // This allocation is actually used only for packing weight and thus will be freed.
        // Still we should be consistent. Fix this.
        Tensor qnnp_weight = _empty_affine_quantized(
            weight_contig.sizes(),
            device(kCPU)
                .dtype(kQUInt8)
                .memory_format(MemoryFormat::ChannelsLast),
            weight_scales_data[0],
            w_zero_points[0],
            nullopt);
        auto* qnnp_w_data = qnnp_weight.template data_ptr<quint8>();
        auto wt_numel = weight_contig.numel();
        for (int i = 0; i < wt_numel; ++i) {
          qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
        }
        Tensor qbias;
        // Original bias was float, so we requantize it here.
        if (conv_p.per_channel) {
          Tensor bias_quant_scales =
              weight_contig.q_per_channel_scales() * act_input_scale;
          Tensor bias_zp = zeros(bias_quant_scales.sizes(), kInt);
          qbias = native::quantize_per_channel_cpu(
              bias_fp32, bias_quant_scales, bias_zp, 0, kQInt32);
        } else {
          qbias = native::quantize_per_tensor(
              bias_fp32,
              weight_contig.q_scale() * act_input_scale,
              0,
              kQInt32);
        }

        // Update the input scale to not pack again.
        input_scale = act_input_scale;
        w.reset();
        w = make_unique<qnnpack::PrePackConvWeights>(
            conv_p,
            w_zero_points.data(),
            reinterpret_cast<u8*>(qnnp_w_data),
            reinterpret_cast<i32*>(qbias.template data_ptr<qint32>()));
        pack_w = w.get();
        if (globalContext().releaseWeightsWhenPrepacking()) {
            // On mobile, we release the original weight by resetting the intrusive_ptr.
            // Calling unpack after this will throw an assertion.
            orig_weight.reset();
        }

        // Set padding buffer to zero point. This can only be done if we want
        // to do it only once.
        if (zero_buffer_size) {
          memset(
              convolution_op->zero_buffer, act_nhwc.q_zero_point(), zero_buffer_size);
        }
      }

      TORCH_INTERNAL_ASSERT(pack_w != nullptr, "Packed Weights are NULL");
      SmallVector<i64, kSpatialDim + 2> output_shape;
      if (transpose()) {
        output_shape = MakeDeConvOutputShape<kSpatialDim>(N, M, {H, W},
            kernel_, stride(), padding(), output_padding(), dilation());
      } else {
        output_shape = MakeConvOutputShape<kSpatialDim>(N, M, {H, W},
            kernel_, stride(), padding(), dilation());
      }

      if (act_nhwc.numel() > 0) {
        TORCH_CHECK(
            all_of(
                output_shape.begin(),
                output_shape.end(),
                [](i64 i) { return i > 0; }),
            "quantized::conv2d (qnnpack): each dimension of output tensor should "
            "be greater than 0.")
      }

      // Allocate output Tensor and a buffer for QNNPACK to use
      Tensor output = native::empty_affine_quantized(
          output_shape,
          kQUInt8,
          nullopt /* layout */,
          kCPU,
          nullopt /* pin_memory */,
          output_scale,
          output_zero_point,
          MemoryFormat::ChannelsLast);

      pytorch_qnnp_status run_status;
      if (transpose()) {
        run_status = qnnpack::qnnpackDeConv(
            conv_p,
            convolution_op.get(),
            pack_w->getPackedWeights(),
            N,
            H,
            W,
            act_nhwc.q_zero_point(),
            reinterpret_cast<u8*>(act_nhwc.template data_ptr<quint8>()),
            w_zero_points.data(),
            requantization_scales.data(),
            output.q_zero_point(),
            output_min,
            output_max,
            reinterpret_cast<u8*>(output.template data_ptr<quint8>()),
            pthreadpool_());
      } else {
        run_status = qnnpack::qnnpackConv(
            conv_p,
            convolution_op.get(),
            pack_w->getPackedWeights(),
            N,
            H,
            W,
            act_nhwc.q_zero_point(),
            reinterpret_cast<u8*>(act_nhwc.template data_ptr<quint8>()),
            w_zero_points.data(),
            requantization_scales.data(),
            output.q_zero_point(),
            output_min,
            output_max,
            reinterpret_cast<u8*>(output.template data_ptr<quint8>()),
            pthreadpool_());
      }

      TORCH_INTERNAL_ASSERT(
          run_status == pytorch_qnnp_status_success,
          "failed to run quantized::conv2d (qnnpack) operator");

      return output;
        */
    }
}



/**
  | FBGEMM uses vpmaddubsw instruction
  | to multiply activations (u8)
  | and weights (i8). https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maddubs_epi16&expand=3284,3530
  | vpmaddubsw operates on a vector of activations
  | and a vector of weights. If these vectors
  | are
  | 
  | A (u8) = a0, a1, a2, a3 ... and
  | 
  | B (i8) = b0, b1, b2, b3 ... the result
  | of this instruction is an i16 vector
  | with values
  | 
  | C (i16) = a0*b0 + a1*b1, a2*b2 + a3*b3
  | ...
  | 
  | For large values of A and/or B the result
  | (a0*b0 + a1*b1) might not fit into an
  | i16 number. So the instruction
  | saturates them to max (or min) possible
  | value of an i16 number. Such behavior
  | is expected for the implementation
  | below.
  | 
  | For example, a0 = 255, a1 = 255, b0 = 127
  | and b1 = 127 the actual result 64770 overflows
  | for an i16 number (-32768, 32767)
  | so the returned result is 32767.
  |
  */
pub struct QConvInt8<const kSpatialDim: i32,const kReluFused: bool> {

}

impl<const kSpatialDim: i32,const kReluFused: bool> QConvInt8<kSpatialDim,kReluFused> {
    
    pub fn run(
        act:               Tensor,
        packed_weight:     &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            if (kReluFused) {
          return packed_weight->apply_relu(act, output_scale, output_zero_point);
        } else {
          return packed_weight->apply(act, output_scale, output_zero_point);
        }
        */
    }
}

pub struct QConv1dInt8<const kReluFused: bool> {

}

impl<const kReluFused: bool> QConv1dInt8<kReluFused> {

    pub fn run(
        act:               Tensor,
        packed_weight:     &IntrusivePtr<ConvPackedParamsBase<2>>,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            Tensor output;
        // N, C, L -> N, C, 1, L
        act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
        if (kReluFused) {
          output = packed_weight->apply_relu(act, output_scale, output_zero_point);
        } else {
          output = packed_weight->apply(act, output_scale, output_zero_point);
        }
        // N, C, 1, L -> N, C, L
        return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
        */
    }
}

/**
  | kernel for maintaining backward compatibility
  |
  */
pub struct QConvInt8ForBC<const kSpatialDim: i32,const kReluFused: bool> {

}

impl<const kSpatialDim: i32,const kReluFused: bool> QConvInt8ForBC<kSpatialDim,kReluFused> {
    
    pub fn run(
        act:               Tensor,
        packed_weight:     &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>,
        stride:            TorchList<i64>,
        padding:           TorchList<i64>,
        dilation:          TorchList<i64>,
        groups:            i64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            if (kReluFused) {
          TORCH_WARN_ONCE(
              "Arguments [stride, padding, dilation, groups] in ops.quantized.conv"
              + to_string(kSpatialDim) + "d_relu, " +
              "have been removed, please update your model to remove these arguments.");
          return packed_weight->apply_relu(act, output_scale, output_zero_point);
        } else {
          TORCH_WARN_ONCE(
              "Arguments [stride, padding, dilation, groups] in ops.quantized.conv"
              + to_string(kSpatialDim) + "d, " +
              "have been removed, please update your model to remove these arguments.");
          return packed_weight->apply(act, output_scale, output_zero_point);
        }
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d"),          QConv1dInt8<false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu"),     QConv1dInt8<true>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"),      QConvInt8<2, false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d.new"),      QConvInt8<3, false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu.new"), QConvInt8<3, true>::run);
      // for backward compatibility
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d"), QConvInt8ForBC<2, false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu"), QConvInt8ForBC<2, true>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d"), QConvInt8ForBC<3, false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu"), QConvInt8ForBC<3, true>::run);

      // transpose
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::conv_transpose3d"),
          QConvInt8<3, false>::run);
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d"),      QConvInt8<2, false>::run);
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_relu"), QConvInt8<2, true>::run);

      // transpose
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
    }
    */
}
