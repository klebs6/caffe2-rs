crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qconv_unpack.cpp]

#[cfg(target_feature = "fbgemm")]
impl<const SPATIAL_DIM: usize> PackedConvWeight<SPATIAL_DIM> {
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto* packed_weights_p = w.get();
      // output channels
      const int output_channels = packed_weights_p->outputChannels();
      const int input_channels = packed_weights_p->inputChannels();
      const int groups = packed_weights_p->groups();

      const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
      // R (kernel height)
      const int kernel_h = kernel[kSpatialDim - 2];
      // S (kernel width)
      const int kernel_w = kernel[kSpatialDim - 1];

      const int C_per_G = input_channels / groups;

      // Tensor for unpacked weights
      // Unpacked format would be physical KRS(C/G) but logical KCRS (channels
      // first) because that's how
      // ChannelsLast3d is not available now.FBGEMM stores the weights
      // TODO: Unify 2d and 3d when ChannelsLast3d is ready.
      Tensor unpacked_weights;
      if (q_scheme == kPerTensorAffine) {
        unpacked_weights = kSpatialDim == 2
            ? _empty_affine_quantized(
                  {output_channels, C_per_G, kernel_h, kernel_w},
                  device(kCPU)
                      .dtype(kQInt8)
                      .memory_format(MemoryFormat::ChannelsLast),
                  w_scale[0],
                  w_zp[0],
                  nullopt)
            : native::fbgemm_utils::
                  MakeEmptyAffineQuantizedChannelsLast3dTensor(
                      output_channels,
                      C_per_G,
                      kernel_d,
                      kernel_h,
                      kernel_w,
                      device(kCPU).dtype(kQInt8),
                      w_scale[0],
                      w_zp[0]);
      } else if (q_scheme == kPerChannelAffine) {
        TORCH_CHECK(
            !transpose(),
            "Per Channel Quantization is currently disabled for transposed conv");
        auto scales = from_blob(
            w_scale.data(), w_scale.size(), device(kCPU).dtype(kFloat));
        auto zero_points = from_blob(
            w_zp.data(), w_zp.size(), device(kCPU).dtype(kInt));
        unpacked_weights = kSpatialDim == 2
            ? _empty_per_channel_affine_quantized(
                  {output_channels, C_per_G, kernel_h, kernel_w},
                  scales.toType(kDouble),
                  zero_points.toType(kLong),
                  0, /* The output channel axis is 0 */
                  device(kCPU).dtype(kQInt8),
                  MemoryFormat::ChannelsLast)
            : native::fbgemm_utils::
                  MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
                      output_channels,
                      C_per_G,
                      kernel_d,
                      kernel_h,
                      kernel_w,
                      device(kCPU).dtype(kQInt8),
                      scales.toType(kDouble),
                      zero_points.toType(kLong));
      } else {
        TORCH_CHECK(false, "Unsupported qscheme: ", toString(q_scheme));
      }
      i8* unpacked_weights_p =
          reinterpret_cast<i8*>(unpacked_weights.data_ptr<qint8>());
      packed_weights_p->unpack(unpacked_weights_p);
      if(transpose()){
        unpacked_weights =
            native::fbgemm_utils::TransposeConvTensorUnpackConversion<
                kSpatialDim>(unpacked_weights, groups);
      }
      return tuple<Tensor, optional<Tensor>>(
          unpacked_weights, bias);
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl<const SPATIAL_DIM: usize> PackedConvWeightsQnnp<SPATIAL_DIM> {
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            TORCH_CHECK(
          kSpatialDim == 2,
          "QNNPACK only supports conv2d_unpack right "
          "now.");
      TORCH_CHECK(
            orig_weight.defined(),
            "Cannot unpack weights. "
            "Call globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
      return tuple<Tensor, optional<Tensor>>(orig_weight, bias);
        */
    }
}

/**
  | QConvPackWeightInt8 expects its input
  | tensor to be in shape [output_channels,
  | kernel_height, kernel_width, input_channels/Groups]
  | 
  | Therefore, the unpacking of packed
  | weight tensor using QConvUnpackWeightsInt8
  | results in a tensor of the same shape.
  |
  */

pub struct QConvUnpackWeightsInt8<const SPATIAL_DIM: usize = 2> {

}

impl QConvUnpackWeightsInt8 {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto& ctx = globalContext();

    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return packed_weight->unpack();
        }
    #endif

    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          TORCH_CHECK(
              kSpatialDim == 2,
              "quantized::conv2d_unpack (qnnpack): QNNPACK only supports Conv2d "
              "now.");
          return packed_weight->unpack();
        }
    #endif

        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::conv2d_unpack ",
            toString(ctx.qEngine()));
        */
    }
}


pub struct QConv1dUnpackWeightsInt8 {

}

impl QConv1dUnpackWeightsInt8 {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<2>>) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto& ctx = globalContext();
        Tensor weight;
        optional<Tensor> bias;
    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          tie(weight, bias) = packed_weight->unpack();
          weight = weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
          return tuple<Tensor, optional<Tensor>>(weight, bias);
        }
    #endif

    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          tie(weight, bias) = packed_weight->unpack();
          Tensor new_weight = weight.clone();
          new_weight = new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
          return tuple<Tensor, optional<Tensor>>(new_weight, bias);
        }
    #endif

        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::conv1d_unpack ",
            toString(ctx.qEngine()));
        */
    }
}


pub struct QConvStride<const SPATIAL_DIM: usize = 2> {

}

impl QConvStride {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> TorchList<i64> {
        
        todo!();
        /*
            return packed_weight->stride();
        */
    }
}


pub struct QConvPadding<const SPATIAL_DIM: usize = 2> {

}

impl QConvPadding {

    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> TorchList<i64> {
        
        todo!();
        /*
            return packed_weight->padding();
        */
    }
}


pub struct QConvOutputPadding<const SPATIAL_DIM: usize = 2> {

}

impl QConvOutputPadding {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> TorchList<i64> {
        
        todo!();
        /*
            return packed_weight->output_padding();
        */
    }
}

pub struct QConvDilation<const SPATIAL_DIM: usize = 2> {

}

impl QConvDilation {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> TorchList<i64> {
        
        todo!();
        /*
            return packed_weight->dilation();
        */
    }
}

pub struct QConvGroups<const SPATIAL_DIM: usize = 2> {

}

impl<const SPATIAL_DIM: usize> QConvGroups<SPATIAL_DIM> {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> i64 {
        
        todo!();
        /*
            return packed_weight->groups();
        */
    }
}

pub struct QConvTranspose<const SPATIAL_DIM: usize = 2> {

}

impl QConvTranspose {
    
    pub fn run(packed_weight: &IntrusivePtr<ConvPackedParamsBase<SpatialDim>>) -> i64 {
        
        todo!();
        /*
            return packed_weight->transpose();
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
      // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
      // We use  conv2d_unpack to be consistent with conv3d_unpack
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_stride"), TORCH_FN(QConvStride<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_padding"), TORCH_FN(QConvPadding<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_dilation"), TORCH_FN(QConvDilation<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_groups"), TORCH_FN(QConvGroups<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_transpose"), TORCH_FN(QConvTranspose<2>::run));

      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_stride"), TORCH_FN(QConvStride<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_padding"), TORCH_FN(QConvPadding<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_dilation"), TORCH_FN(QConvDilation<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_groups"), TORCH_FN(QConvGroups<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_transpose"), TORCH_FN(QConvTranspose<3>::run));

      // ConvTranspose is the same, however, we want to have different name.
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_stride"), TORCH_FN(QConvStride<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_padding"), TORCH_FN(QConvPadding<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dilation"), TORCH_FN(QConvDilation<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_groups"), TORCH_FN(QConvGroups<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_transpose"), TORCH_FN(QConvTranspose<2>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_stride"), TORCH_FN(QConvStride<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_padding"), TORCH_FN(QConvPadding<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dilation"), TORCH_FN(QConvDilation<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_groups"), TORCH_FN(QConvGroups<3>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_transpose"), TORCH_FN(QConvTranspose<3>::run));
    }
    */
}
