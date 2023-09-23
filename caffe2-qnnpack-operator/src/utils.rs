crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack_utils.h]

#[cfg(USE_PYTORCH_QNNPACK)]
mod qnnpack {
    use super::*;

    pub struct QnnpackOperatorDeleter {

    }

    impl QnnpackOperatorDeleter {
        
        pub fn invoke(&mut self, op: PyTorchQnnpOperator)  {
            
            todo!();
            /*
                pytorch_qnnp_delete_operator(op);
            */
        }
    }


    pub struct PackedConvWeightsQnnp<const SPATIAL_DIM: usize = 2> {
        base:                  ConvPackedParamsBase<SpatialDim>,
        convolution_op:        Box<PyTorchQnnpOperator,QnnpackOperatorDeleter>,
        w:                     Box<QnnPackPrePackConvWeights>,
        orig_weight:           Tensor,
        bias:                  Tensor,
        stride:                TorchList<i64>,
        padding:               TorchList<i64>,
        output_padding:        TorchList<i64>,
        dilation:              TorchList<i64>,
        groups:                i64,
        transpose:             bool,
        input_scale:           Option<f64>,
        kernel:                Vec<i64>,
        w_scales:              Tensor,
        w_zero_points:         Vec<u8>,
        requantization_scales: Vec<f32>,
        conv_p:                QnnPackConvParam,
        zero_buffer_size:      usize,
    }

    impl PackedConvWeightsQnnp {
        
        pub fn new(
            w:              Box<QnnPackPrePackConvWeights>,
            orig_weight:    Tensor,
            bias:           Tensor,
            stride:         TorchList<i64>,
            padding:        TorchList<i64>,
            output_padding: TorchList<i64>,
            dilation:       TorchList<i64>,
            groups:         i64,
            transpose:      bool,
            input_scale:    Option<f64>,
            kernel:         Vec<i64>,
            w_scale:        Tensor,
            w_zps:          Vec<u8>,
            is_per_channel: bool) -> Self {
        
            todo!();
            /*


                : w(move(w)),
                orig_weight(move(orig_weight)),
                bias(move(bias)),
                stride_(move(stride)),
                padding_(move(padding)),
                output_padding_(move(output_padding)),
                dilation_(move(dilation)),
                groups_(groups),
                transpose_(transpose),
                input_scale(input_scale),
                kernel_(move(kernel)),
                w_scales(w_scale),
                w_zero_points(move(w_zps)),
                conv_p(
                    {(u32)kernel_[1], (u32)kernel_[0]},
                    {(u32)stride_[1], (u32)stride_[0]},
                    {(u32)dilation_[1], (u32)dilation_[0]},
                    {(u32)padding_[0], (u32)padding_[1],
                     (u32)padding_[0], (u32)padding_[1]},
                    {(u32)output_padding_[1], (u32)output_padding_[0]},
                    groups_,
                    transpose ? this->orig_weight.size(0)
                              : this->orig_weight.size(1) * groups_,
                    transpose ? this->orig_weight.size(1) * groups_
                              : this->orig_weight.size(0),
                    transpose_,
                    is_per_channel) 

                  if (conv_p.per_channel && conv_p.ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
                    TORCH_INTERNAL_ASSERT(
                      "Per channel quantized weights are not supported for XZP kernels");
                  }

                  pytorch_qnnp_operator_t convolution{nullptr};
                  // Initially all the params are set to zero.
                  convolution =
                      static_cast<pytorch_qnnp_operator_t>(calloc(1, sizeof(struct pytorch_qnnp_operator)));
                  if (convolution == nullptr) {
                    TORCH_INTERNAL_ASSERT(
                        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
                        sizeof(struct pytorch_qnnp_operator));
                  }

                  convolution_op =
                    unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(convolution);

                  convolution->ukernel_type = conv_p.ukernel_type;
                  convolution->groups = groups;
                  convolution->group_input_channels = conv_p.group_input_channels;
                  convolution->kernel_height = conv_p.kernel_dims[1];
                  convolution->kernel_width = conv_p.kernel_dims[0];
                  convolution->stride_height = conv_p.stride_dims[1];
                  convolution->stride_width = conv_p.stride_dims[0];
                  convolution->dilation_height = conv_p.dilation[1];
                  convolution->dilation_width = conv_p.dilation[0];
                  convolution->input_padding_top = conv_p.padding[0];
                  convolution->input_padding_left = conv_p.padding[1];
                  convolution->input_padding_bottom = conv_p.padding[2];
                  convolution->input_padding_right = conv_p.padding[3];

                  // const usize group_input_channels = conv_p.group_input_channels;
                  const u32 kr = pytorch_qnnp_params.q8conv.kr;
                  const usize k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

                  usize zero_size = sizeof(u8) * k_stride;
                  usize zero_offset = 0;

                  if (transpose_) {
                    convolution->adjustment_width = conv_p.adjustment_dims[0];
                    convolution->adjustment_height = conv_p.adjustment_dims[1];

                    // const u32 kr = pytorch_qnnp_params.q8conv.kr;
                    // const usize k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

                    if (conv_p.group_input_channels < 8) {
                      zero_size += 8;
                      zero_offset = 8;
                    }
                  } else {
                    const bool any_padding = (conv_p.padding[0]| conv_p.padding[1]
                        |conv_p.padding[2] | conv_p.padding[3]) != 0;

                    zero_buffer_size = 0;
                    if (any_padding) {
                      zero_size = 0;
                      zero_offset = 0;
                      if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_dwconv) {
                        const u32 cr = pytorch_qnnp_params.q8dw9.cr;
                        const usize group_stride = (groups + (cr - 1)) & -cr;
                        if (groups >= 8) {
                          zero_size = sizeof(u8) * group_stride;
                          zero_offset = 0;
                        } else {
                          zero_size = sizeof(u8) * group_stride + 8;
                          zero_offset = sizeof(u8) * 8;
                        }
                      } else if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_conv ||
                          conv_p.ukernel_type == pytorch_qnnp_ukernel_type_gemm) {
                        if (conv_p.group_input_channels >= 8) {
                          zero_size = sizeof(u8) * k_stride;
                          zero_offset = 0;
                        } else {
                          zero_size = sizeof(u8) * k_stride + 8;
                          zero_offset = 8;
                        }
                      }
                    }
                  }

                  void* zero_buffer = malloc(zero_size);
                  if (zero_buffer == NULL) {
                    pytorch_qnnp_delete_operator(convolution);
                    pytorch_qnnp_log_error(
                        "failed to allocate %zu bytes for zero padding", zero_size);
                  }
                  // Need to set to input zero point
                  // memset(zero_buffer, input_zero_point, zero_size);
                  zero_buffer_size = zero_size;
                  convolution->zero_buffer = zero_buffer;
                  convolution->zero_pointer =
                    (void*)((uintptr_t)zero_buffer + zero_offset);
            */
        }
        
        pub fn apply(&mut self, 
            input:             &Tensor,
            output_scale:      f64,
            output_zero_point: i64) -> Tensor {
            
            todo!();
            /*
            
            */
        }
        
        pub fn apply_relu(&mut self, 
            input:             &Tensor,
            output_scale:      f64,
            output_zero_point: i64) -> Tensor {
            
            todo!();
            /*
            
            */
        }
        
        pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
            
            todo!();
            /*
            
            */
        }
        
        pub fn prepack(
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
            
            */
        }
        
        pub fn stride(&self) -> TorchList<i64> {
            
            todo!();
            /*
                return stride_;
            */
        }
        
        pub fn padding(&self) -> TorchList<i64> {
            
            todo!();
            /*
                return padding_;
            */
        }
        
        pub fn output_padding(&self) -> TorchList<i64> {
            
            todo!();
            /*
                return output_padding_;
            */
        }
        
        pub fn dilation(&self) -> TorchList<i64> {
            
            todo!();
            /*
                return dilation_;
            */
        }
        
        pub fn groups(&self) -> i64 {
            
            todo!();
            /*
                return groups_;
            */
        }
        
        pub fn transpose(&self) -> bool {
            
            todo!();
            /*
                return transpose_;
            */
        }
        
        
        pub fn apply_impl<const ReluFused: bool>(&mut self, 
            input:             &Tensor,
            output_scale:      f64,
            output_zero_point: i64) -> Tensor {
        
            todo!();
            /*
            
            */
        }
    }

    #[repr(u8)]
    pub enum Activation { 
        NONE = 0, 
        RELU = 1 
    }

    #[cfg(all(__ANDROID__,not(__NDK_MAJOR__)))]
    #[inline] pub fn round<T>(x: f32) -> f32 {

        todo!();
            /*
                return ::nearbyintf(x);
            */
    }

    #[cfg(all(__ANDROID__,not(__NDK_MAJOR__)))]
    #[inline] pub fn round(x: f64) -> f64 {
        
        todo!();
            /*
                return ::nearbyint(x);
            */
    }

    #[cfg(not(all(__ANDROID__,not(__NDK_MAJOR__))))]
    #[inline] pub fn round<T>(x: T) -> T {

        todo!();
            /*
                return nearbyint(x);
            */
    }

    #[inline] pub fn quantize_uint8(
            scale:      f32,
            zero_point: i32,
            value:      f32) -> u8 {
        
        todo!();
            /*
                const i32 qmin = u8::min;
          const i32 qmax = u8::max;
          auto r = zero_point + static_cast<i32>(Round(value / scale));
          r = max(r, qmin);
          r = min(r, qmax);
          return static_cast<u8>(r);
            */
    }

    #[inline] pub fn activation_limits(
            scale:      f32,
            zero_point: i32,
            ac:         Activation) -> (u8,u8) {
        
        todo!();
            /*
                switch (Ac) {
            case Activation::NONE:
              return {u8::min,
                      u8::max};
            case Activation::RELU:
              return {QuantizeUint8(scale, zero_point, 0.0),
                      u8::max};
            default:
        #ifdef _MSC_VER
              __assume(0);
        #else
              __builtin_unreachable();
        #endif
          }
            */
    }

    pub fn qnnpack_avg_pool2d(
            input:             Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
            
            */
    }



    pub fn generate_requantization_scales(
            weight_scales:  &Tensor,
            input_scale:    f32,
            output_scale:   f32,
            requant_scales: &mut Vec<f32>) -> Vec<f32> {
        
        todo!();
            /*
                // Since weight scale is allocated with padding
          // weight_scales.numel() gives us padded num elements.
          const auto num_output_channels_padded = weight_scales.numel();
          float *const weight_scales_data = weight_scales.data_ptr<float>();
          if (static_cast<i64>(requant_scales.size()) < num_output_channels_padded) {
            requant_scales.resize(num_output_channels_padded);
          }
          for (const auto i : irange(num_output_channels_padded)) {
            const auto inverse_output_scale = 1.f /output_scale;
            requant_scales[i] = (weight_scales_data[i] * input_scale) * inverse_output_scale;
            TORCH_CHECK(
                (requant_scales[i] > 0.0f && isnormal(requant_scales[i])),
                "failed to create op with requantization scale: ",
                requant_scales[i],
                ": requantization scale must be finite and positive");
          }
          return requant_scales;
            */
    }


    pub fn make_zero_points_and_scales_tensor(
            weight_contig: &Tensor,
            transpose:     bool,
            groups:        u32) -> (Vec<u8>,Tensor) {
        let transpose: bool = transpose.unwrap_or(false);
        let groups: u32 = groups.unwrap_or(1);

        todo!();
            /*
                const int out_ch_idx = transpose ? 1 : 0;
          const auto num_output_channels = weight_contig.size(out_ch_idx) * (transpose ? groups : 1);
          // Add 8 to account for bufferring needed by QNNPACK.
          const auto num_output_channels_padded = num_output_channels + 8;
          const auto qtype = weight_contig.qscheme();
          vector<u8> weight_zp(num_output_channels_padded, 0);
          // Adjust weight zero point, similar to weight data.
          if (qtype == kPerTensorAffine) {
            for (const auto i : irange(num_output_channels)) {
              weight_zp[i] = (u8)(weight_contig.q_zero_point() + 128);
            }
          } else if (qtype == kPerChannelAffine) {
            TORCH_CHECK(
                weight_contig.q_per_channel_zero_points().scalar_type() == kLong,
                "Per channel zero points dtype must be long int.");
            const i64* per_channel_zero_points =
              weight_contig.q_per_channel_zero_points().data_ptr<i64>();
            for (const auto i : irange(num_output_channels)) {
              weight_zp[i] = (u8)(per_channel_zero_points[i] + 128);
            }
          } else {
            TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
          }
           Tensor weight_scales =
            empty(
                {num_output_channels_padded},
                device(kCPU).dtype(kFloat));
          float *const weight_scales_data = weight_scales.data_ptr<float>();
          if (qtype == kPerTensorAffine) {
            for (const auto i : irange(num_output_channels)) {
              weight_scales_data[i] = weight_contig.q_scale();
            }
          } else if (qtype == kPerChannelAffine) {
            TORCH_CHECK(
                weight_contig.q_per_channel_scales().scalar_type() == kDouble,
                "Per channel scales dtype must be double.");
            const double *const per_channel_scales =
              weight_contig.q_per_channel_scales().data_ptr<double>();
            for (const auto i : irange(num_output_channels)) {
              weight_scales_data[i] = static_cast<float>(per_channel_scales[i]);
            }
          } else {
            TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
          }
          for (const auto i : irange(num_output_channels, num_output_channels_padded)) {
            weight_scales_data[i] = 1.f;
          }
          return {weight_zp, weight_scales};
            */
    }
}
