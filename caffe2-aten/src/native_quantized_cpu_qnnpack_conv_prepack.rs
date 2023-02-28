crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/conv-prepack.cc]

impl PrePackConvWeights {
    
    pub fn new(
        conv_p:             &ConvParam,
        kernel_zero_points: *const u8,
        kernel:             *const u8,
        bias:               *const i32) -> Self {
    
        todo!();
        /*


            output_channels_ = conv_p.output_channels;
      enum pytorch_qnnp_ukernel_type ukernel_type = conv_p.ukernel_type;
      const u32 kernel_width = conv_p.kernel_dims[0];
      const u32 kernel_height = conv_p.kernel_dims[1];
      const u32 groups = conv_p.groups;

      if (conv_p.transpose && ukernel_type != pytorch_qnnp_ukernel_type_conv) {
        pytorch_qnnp_log_error("Wrong micro-kernel for deconvolution");
        assert("QNNPACK Runtime Error.");
      }

      const usize kernel_size = kernel_height * kernel_width;
      switch (ukernel_type) {
        case pytorch_qnnp_ukernel_type_dwconv: {
          const u32 cr = pytorch_qnnp_params.q8dw9.cr;
          const u32 c_stride = (groups + (cr - 1)) & -cr;
          const usize packed_weights_size =
              (sizeof(u8) * kernel_size + sizeof(i32)) * c_stride;
          packed_weights_ = malloc(packed_weights_size);
          if (packed_weights_ == nullptr) {
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for packed weights",
                packed_weights_size);
            assert("QNNPACK Runtime Error.");
          }

          switch (kernel_size) {
            case 9:
              pytorch_pack_q8dw_wrq(
                  kernel_height,
                  kernel_width,
                  groups,
                  cr,
                  kernel,
                  bias,
                  packed_weights_);
              break;
            case 25:
              /* change this later */
              pytorch_pack_q8dw_w_dilation(
                  kernel_height,
                  kernel_width,
                  groups,
                  cr,
                  0,
                  kernel_height,
                  0,
                  2,
                  kernel,
                  bias,
                  packed_weights_,
                  true);
              pytorch_pack_q8dw_w_dilation(
                  kernel_height,
                  kernel_width,
                  groups,
                  cr,
                  0,
                  kernel_height,
                  2,
                  4,
                  kernel,
                  bias,
                  (char*)packed_weights_ +
                      (10 + sizeof(i32) / sizeof(u8)) * c_stride,
                  false);
              pytorch_pack_q8dw_w_dilation(
                  kernel_height,
                  kernel_width,
                  groups,
                  cr,
                  0,
                  kernel_height,
                  4,
                  5,
                  kernel,
                  bias,
                  (char*)packed_weights_ +
                      (20 + sizeof(i32) / sizeof(u8)) * c_stride,
                  false);
              break;
            default:
              PYTORCH_QNNP_UNREACHABLE;
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_xzp_gemm: {
          const u32 nr = pytorch_qnnp_params.q8conv_xzp.nr;
          const u32 kr = pytorch_qnnp_params.q8conv_xzp.kr;
          const u32 sr = pytorch_qnnp_params.q8conv_xzp.kc;
          const u32 n_stride = (conv_p.group_output_channels + (nr - 1)) & -nr;
          const u32 k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

          const usize packed_group_weights_size =
              (sizeof(u8) * kernel_size * k_stride + sizeof(i32)) *
              n_stride;
          packed_weights_ = malloc(packed_group_weights_size * groups);
          if (packed_weights_ == nullptr) {
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for packed weights",
                packed_group_weights_size * groups);
            assert("QNNPACK Runtime Error.");
          }
          /* The XZP ukernel needs the padding to be 0 */
          memset(packed_weights_, 0, packed_group_weights_size * groups);

          for (u32 group = 0; group < groups; group++) {
            pytorch_pack_swizzle_q8gemm_brq(
                conv_p.group_output_channels,
                conv_p.group_input_channels,
                nr,
                kr,
                sr,
                kernel +
                    group * conv_p.group_output_channels *
                        conv_p.group_input_channels,
                bias + group * conv_p.group_output_channels,
                (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_gemm:
        case pytorch_qnnp_ukernel_type_conv: {
          const u32 nr = pytorch_qnnp_params.q8conv.nr;
          const u32 kr = pytorch_qnnp_params.q8conv.kr;
          const u32 n_stride = (conv_p.group_output_channels + (nr - 1)) & -nr;
          const u32 k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

          const usize packed_group_weights_size =
              (sizeof(u8) * kernel_size * k_stride + sizeof(i32)) *
              n_stride;
          packed_weights_ = malloc(packed_group_weights_size * groups);
          if (packed_weights_ == nullptr) {
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for packed weights",
                packed_group_weights_size * groups);
            assert("QNNPACK Runtime Error.");
          }
          // We likely won't needs this once packing functions are appropriately
          // modified. Remove it then.
          memset(
              packed_weights_,
              kernel_zero_points[0],
              packed_group_weights_size * groups);

          switch (ukernel_type) {
            case pytorch_qnnp_ukernel_type_gemm:
              for (u32 group = 0; group < groups; group++) {
                pytorch_pack_q8gemm_wrq(
                    conv_p.group_output_channels,
                    conv_p.group_input_channels,
                    nr,
                    nr,
                    kr,
                    kernel +
                        group * conv_p.group_output_channels *
                            conv_p.group_input_channels,
                    bias + group * conv_p.group_output_channels,
                    kernel_zero_points + group * conv_p.group_output_channels,
                    (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
              }
              break;
            case pytorch_qnnp_ukernel_type_conv:  // The transpose can only be here
              for (u32 group = 0; group < groups; group++) {
                const u8* const kernel_p = kernel
                  + group * conv_p.group_output_channels * kernel_size
                  * conv_p.group_input_channels;
                const i32* const bias_p = bias
                  + group * conv_p.group_output_channels;
                if (conv_p.transpose) {  // Note that only runtime packing is here
                  pytorch_pack_q8deconv_wrq(
                      conv_p.group_output_channels,
                      kernel_size,
                      conv_p.group_input_channels,
                      nr,
                      kr,
                      kernel_p,
                      bias_p,
                      kernel_zero_points + group * conv_p.group_output_channels,
                      (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
                } else {
                  pytorch_pack_q8conv_wrq(
                      conv_p.group_output_channels,
                      kernel_size,
                      conv_p.group_input_channels,
                      nr,
                      kr,
                      kernel_p,
                      bias_p,
                      kernel_zero_points + group * conv_p.group_output_channels,
                      (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
                }
              }
              break;
            default:
              PYTORCH_QNNP_UNREACHABLE;
          }
          break;
        }
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }
        */
    }
}

