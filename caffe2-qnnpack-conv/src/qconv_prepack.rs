crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qconv_prepack.cpp]

#[cfg(feature = "fbgemm")]
impl<const SPATIAL_DIM: usize> PackedConvWeight<SPATIAL_DIM> {
    
    pub fn prepack(&mut self, 
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      bool) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
        
        todo!();
        /*
            TORCH_CHECK(
          weight.ndimension() == kSpatialDim + 2,
          "Weights are expected to have ",
          kSpatialDim + 2,
          " dimensions");
      TORCH_CHECK(
          stride.size() == kSpatialDim,
          "stride should contain ",
          kSpatialDim,
          " elements for ",
          kSpatialDim,
          "D convolution.");
      TORCH_CHECK(
          padding.size() == kSpatialDim,
          "Specify front/top/left padding only. "
          "end/bottom/right padding assumed to be equal to front/top/left");
      TORCH_CHECK(
          !transpose || output_padding.size() == kSpatialDim,
          "quantized::conv_prepack: Specify top/left output padding "
          "only. bottom/right padding assumed to be equal to top/left");
      TORCH_CHECK(
          dilation.size() == kSpatialDim,
          "dilation should contain ",
          kSpatialDim,
          " elements for ",
          kSpatialDim,
          "D convolution.");
      const int input_channels = transpose ? weight.size(0)
                                           : weight.size(1) * groups;
      const int output_channels = transpose ? weight.size(1) * groups
                                            : weight.size(0);
      const int kernel_d = kSpatialDim == 2 ? 1 : weight.size(2);
      const int kernel_h = weight.size(kSpatialDim);
      const int kernel_w = weight.size(kSpatialDim + 1);

      // mini-batch doesn't have any impact on how we pack weights
      // so we pass it as 1
      // Input image height/width also don't have any impact on how we pack
      // weights so we can pass any values
      const fbgemm::conv_param_t<kSpatialDim> conv_p =
          native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
              1, // dummy batch size
              input_channels,
              output_channels,
              kSpatialDim == 2 ? vector<int>{28, 28} // dummy image size
                               : vector<int>{28, 28, 28},
              groups,
              kSpatialDim == 2 ? vector<int>{kernel_h, kernel_w}
                               : vector<int>{kernel_d, kernel_h, kernel_w},
              vector<int>(stride.begin(), stride.end()),
              vector<int>(padding.begin(), padding.end()),
              vector<int>(dilation.begin(), dilation.end()),
              vector<int>(output_padding.begin(), output_padding.end()),
              transpose);

      const auto qtype = weight.qscheme();
      vector<i32> zero_points;
      if (qtype == kPerTensorAffine) {
        zero_points = {static_cast<i32>(weight.q_zero_point())};
      } else if (qtype == kPerChannelAffine) {
        TORCH_CHECK(
            !transpose,
            "Per Channel Quantization is currently disabled for transposed conv");
        zero_points.resize(output_channels);
        for (int i = 0; i < output_channels; ++i) {
          zero_points[i] = weight.q_per_channel_zero_points()[i].item<i32>();
        }
      } else {
        TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
      }

      // FBGEMM expects weights to be in channels last
      // TODO: Change this when ChannelsLast3d is ready.
      // FBGEMM needs G OC/G kDim0 ... kDimN IC/G
      // for both conv and conv transpose
      // but PyTorch lays them out as {out_c, in_c/groups, kH, kW}
      // (or for ConvTranspose {in_c, out_c/groups, kH, kW})
      const Tensor weight_nhwc =
          native::fbgemm_utils::ConvertConvWeightsToChannelLastTensor<kSpatialDim>(weight, groups, transpose);
      const i8* weight_data_int8 =
              reinterpret_cast<i8*>(weight_nhwc.data_ptr<qint8>());
      vector<i32> col_offsets(output_channels);
      // compute column offsets (Similar to
      // fbgemm::col_offsets_with_zero_pt_s8acc32_ref) please note that offsets
      // include the sum of columns as well as the scalar term weight_zero_point *
      // KDim
      const int input_channels_per_group = input_channels / groups;
      const int output_channels_per_group = output_channels / groups;
      const int inner_size =
          kernel_d * kernel_h * kernel_w * input_channels_per_group;
      for (const auto g : irange(groups)) {
        for (int i = 0; i < output_channels_per_group; ++i) {
          const int c = g * output_channels_per_group + i;
          i32 sum = 0;
          for (int j = 0; j < inner_size; ++j) {
            sum += static_cast<i32>(weight_data_int8[c * inner_size + j]);
          }
          if (qtype == kPerTensorAffine) {
            col_offsets[c] = sum - zero_points[0] * inner_size;
          } else {
            col_offsets[c] = sum - zero_points[c] * inner_size;
          }
        }
      }

      vector<float> scales;
      if (qtype == kPerTensorAffine) {
        scales = {static_cast<float>(weight.q_scale())};
      } else if (qtype == kPerChannelAffine) {
        scales.resize(output_channels);
        for (int i = 0; i < output_channels; ++i) {
          scales[i] = weight.q_per_channel_scales()[i].item<float>();
        }
      }

      optional<Tensor> bias_contig;
      if (bias.has_value()) {
        Tensor bias_vec = bias.value();
        TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias_vec.size(0) == output_channels,
            "bias should have K elements: " + to_string(output_channels));
        bias_contig = bias->contiguous();
      }

      auto ret_ptr = make_intrusive<PackedConvWeight<kSpatialDim>>(
          PackedConvWeight<kSpatialDim>{
              make_unique<fbgemm::PackWeightsForConv<kSpatialDim>>(
                  conv_p, weight_data_int8),
              bias_contig,
              stride,
              padding,
              output_padding,
              dilation,
              groups,
              transpose,
              col_offsets,
              kSpatialDim == 2 ? vector<i64>{kernel_h, kernel_w}
                               : vector<i64>{kernel_d, kernel_h, kernel_w},
              scales,
              zero_points,
              qtype});

      return ret_ptr;
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl PackedConvWeightsQnnp {
    
    pub fn prepack<const kSpatialDim: i32>(&mut self, 
        weight:         Tensor,
        bias_in:        Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      bool) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
    
        todo!();
        /*
            TORCH_CHECK(
          kSpatialDim == 2 || kSpatialDim == 3,  // 1D is packed as 2d, hence we don't need other checks
          "QNNPACK packing only supports 2D / 3D convolution.");
      TORCH_CHECK(
          weight.ndimension() == kSpatialDim + 2,
          "quantized::conv_prepack (qnnpack): Weights are expected to have ",
          kSpatialDim + 2, " dimensions, found shape ", weight.sizes());
      TORCH_CHECK(
          stride.size() == kSpatialDim,
          "quantized::conv_prepack (qnnpack): ",
          kSpatialDim, "D convolution expects stride to have ",
          kSpatialDim, " elements.");
      TORCH_CHECK(
          padding.size() == kSpatialDim,
          "quantized::conv_prepack (qnnpack): Specify top/left input padding "
          "only. bottom/right padding assumed to be equal to top/left");
      TORCH_CHECK(
          !transpose || output_padding.size() == kSpatialDim,
          "quantized::conv_prepack (qnnpack): Specify top/left output padding "
          "only. bottom/right padding assumed to be equal to top/left");
      TORCH_CHECK(
          dilation.size() == kSpatialDim,
          "quantized::conv_prepack (qnnpack): ",
          kSpatialDim, "D convolution expects dilation to have ",
          kSpatialDim, " elements.");

      native::initQNNPACK();

      // QNNPACK expects weights to be of the format {out_c, kH, kW, in_c/groups},
      // but PyTorch lays them out as {out_c, in_c/groups, kH, kW}
      // (or for ConvTranspose {in_c, out_c/groups, kH, kW})
      const usize out_ch = transpose ? weight.size(1) * groups : weight.size(0);
      const u32 kernel_h = weight.size(2);
      const u32 kernel_w = weight.size(3);

      Tensor bias_fp32;
      if (bias_in.has_value()) {
        bias_fp32 = bias_in.value();
      } else {
        bias_fp32 = zeros(out_ch, weight.options().dtype(kFloat));
      }

      TORCH_CHECK(
          !bias_fp32.defined() ||
              (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
          "quantized::conv2d_prepack (qnnpack): expected bias to be 1-dimensional "
          "with ",
          out_ch,
          " elements",
          ", but got bias of size ",
          bias_fp32.sizes(),
          " instead. "
          "(weight dimensions: ",
          weight.sizes(), " , transpose: ",
          (transpose ? "True)." : "False).")
      );

      TORCH_CHECK(
          !bias_fp32.defined() ||
              (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
          "quantized::conv3d_prepack (qnnpack): expected bias to be 1-dimensional "
          "with ",
          out_ch,
          " elements",
          ", but got bias of size ",
          bias_fp32.sizes(),
          " instead. "
          "(weight dimensions: ",
          weight.sizes(), " , transpose: ",
          (transpose ? "True)." : "False).")
      );

      auto weight_contig = weight.contiguous(MemoryFormat::ChannelsLast);
      const bool is_per_channel = weight_contig.qscheme() == kPerChannelAffine;

      vector<u8> w_zero_points;
      Tensor w_scales;
      tie(w_zero_points, w_scales) =
          make_zero_points_and_scales_tensor(weight_contig, transpose, groups);
      // We set the pre-packed conv weights to nullptr below as we call pre-pack
      // during the first invocation of operator run. Refer to qconv.cpp for more
      // details. TODO Update to actually call pre-pack here once bias is removed
      // from pre-packing step.
      intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> ret_ptr =
          make_intrusive<PackedConvWeightsQnnp<kSpatialDim>>(
              PackedConvWeightsQnnp<kSpatialDim>{
                  nullptr, /* PrePackConvWeights */
                  weight_contig, /* i8 weight */
                  bias_fp32.contiguous(), /* fp32 bias */
                  stride,
                  padding,
                  output_padding,
                  dilation,
                  groups,
                  transpose,
                  nullopt, /* input_scale */
                  {kernel_h, kernel_w},
                  w_scales,
                  move(w_zero_points),
                  is_per_channel});

      return ret_ptr;
        */
    }
}

pub struct QConvPackWeightInt8<const SPATIAL_DIM: usize = 2> {

}

impl QConvPackWeightInt8 {
    
    pub fn run_conv(
        weight:   Tensor,
        bias:     Option<Tensor>,
        stride:   TorchList<i64>,
        padding:  TorchList<i64>,
        dilation: TorchList<i64>,
        groups:   i64) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
        
        todo!();
        /*
            TorchList<i64> output_padding;
        output_padding.reserve(kSpatialDim);
        for (int idx = 0; idx < kSpatialDim; ++idx) {
          output_padding.push_back((i64)0);
        }
        return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                    /*transpose=*/false);
        */
    }
    
    pub fn run_deconv(
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
        
        todo!();
        /*
            return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                    /*transpose=*/true);
        */
    }
    
    pub fn run(
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      bool) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
        
        todo!();
        /*
            auto& ctx = globalContext();
    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return PackedConvWeight<kSpatialDim>::prepack(
              weight, bias, stride, padding, output_padding, dilation, groups,
              transpose);
        }
    #endif

    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          TORCH_CHECK(
              kSpatialDim == 2,
              "quantized::conv_prepack (qnnpack): QNNPACK only supports Conv1d "
              "and Conv2d now.");
          return PackedConvWeightsQnnp<kSpatialDim>::prepack(
              weight, bias, stride, padding, output_padding, dilation, groups,
              transpose);
        }
    #endif

        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::conv2d_prepack ",
            toString(ctx.qEngine()));
        */
    }
}

pub struct QConv1dPackWeightInt8 {

}

impl QConv1dPackWeightInt8 {
    
    pub fn run_conv(
        weight:   Tensor,
        bias:     Option<Tensor>,
        stride:   TorchList<i64>,
        padding:  TorchList<i64>,
        dilation: TorchList<i64>,
        groups:   i64) -> IntrusivePtr<ConvPackedParamsBase<2>> {
        
        todo!();
        /*
            const TorchList<i64> output_padding({0});
        return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                    /*transpose=*/false);
        */
    }
    
    pub fn run_deconv(
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64) -> IntrusivePtr<ConvPackedParamsBase<2>> {
        
        todo!();
        /*
            return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                    /*transpose=*/true);
        */
    }
    
    pub fn run(
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      bool) -> IntrusivePtr<ConvPackedParamsBase<2>> {
        
        todo!();
        /*
            auto& ctx = globalContext();
        if (weight.dim() == 3) {
          weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
        }
        stride = quant_utils::MakeArgForConv1d(stride, 1);
        padding = quant_utils::MakeArgForConv1d(padding, 0);
        output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
        dilation = quant_utils::MakeArgForConv1d(dilation, 1);
    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return PackedConvWeight<2>::prepack(
              weight, bias, stride, padding, output_padding, dilation, groups,
              transpose);
        }
    #endif

    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          return PackedConvWeightsQnnp<2>::prepack(
              weight, bias, stride, padding, output_padding, dilation, groups,
              transpose);
        }
    #endif
        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::conv1d_prepack ",
            toString(ctx.qEngine()));
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      // Conv
      // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_conv));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
      // ConvTranspose
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
      // Conv
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_conv));
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_conv));
      // ConvTranspose
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d_prepack"), TORCH_FN(QConv1dPackWeightInt8::run_deconv));
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d_prepack"), TORCH_FN(QConvPackWeightInt8<2>::run_deconv));
      m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose3d_prepack"), TORCH_FN(QConvPackWeightInt8<3>::run_deconv));
    }
    */
}

