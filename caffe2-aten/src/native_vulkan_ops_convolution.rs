crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Convolution.h]

pub enum Conv2dMethod {
    Conv2dDepthwise,
    Conv2dPointwise,
    Conv2dOld,
    Conv2dSlidingWindow,
    Conv2dWinograd_2_3,
}

pub struct Conv2dOpContextPacked {
    v_weight:   VTensor,
    v_bias:     VTensor,
    filter:     [i64; 4],
    stride:     [i64; 2],
    padding:    [i64; 2],
    dilation:   [i64; 2],
    groups:     i32,
    output_min: f32,
    output_max: f32,
}

pub struct Conv2dOpContextUnpacked {
    weight:     Tensor,
    bias:       Option<Tensor>,
    filter:     Vec<i64>,
    stride:     Vec<i64>,
    padding:    Vec<i64>,
    dilation:   Vec<i64>,
    groups:     i64,
    output_min: Option<Scalar>,
    output_max: Option<Scalar>,
}

pub struct Conv2dOpContext {
    base:     TorchJitCustomClassHolder,
    packed:   Conv2dOpContextPacked,
    unpacked: Conv2dOpContextUnpacked,
    method:   Conv2dMethod,
}

impl HasState for Conv2dOpContext {

    type State = (
        Tensor,
        Option<Tensor>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        i64,
        Option<Scalar>,
        Option<Scalar>
    );
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Convolution.cpp]

pub mod experimentation {
    pub const USE_CONV2D_OLD_API: bool = false;
    pub const USE_WINOGRAD_CONVS: bool = false;
}

#[inline] pub fn is_depthwise(
    filter: &[i32],
    groups: i64) -> bool {
    
    todo!();
        /*
            return (filter[Layout::Filter::output] == groups) &&
             // Only K == 1 supported.
             (filter[Layout::Filter::input] == 1);
        */
}

#[inline] pub fn is_pointwise(filter: &[i32]) -> bool {
    
    todo!();
        /*
            return (1 == filter[Layout::Filter::height]) &&
             (1 == filter[Layout::Filter::width]);
        */
}

pub fn all_lessthan(
    arr: &[i32],
    t:   i32) -> bool {
    
    todo!();
        /*
            bool retval = true;
      for (usize i = 0; i < arr.size(); i++) {
        retval = retval && (arr[i] < t);
      }
      return retval;
        */
}

#[inline] pub fn is_winograd_n_3(
        filter:   &[i32],
        stride:   &[i32],
        dilation: &[i32]) -> bool {
    
    todo!();
        /*
            return (3 == filter[Layout::Filter::height]) &&
             (3 == filter[Layout::Filter::width]) &&
             all_lessthan(stride, 2) &&
             all_lessthan(dilation, 2);
        */
}

pub fn determine_method(
    filter:   &[i32],
    stride:   &[i32],
    padding:  &[i32],
    dilation: &[i32],
    groups:   i64) -> Conv2dMethod {

    todo!();
        /*
            if (is_depthwise(filter, groups))
        return Conv2dDepthwise;
      if (Experimentation::kUseConv2dOldApi)
        return Conv2dOld;
      if (is_pointwise(filter))
        return Conv2dPointwise;
      if (Experimentation::kUseWinogradConvs && is_winograd_n_3(filter, stride, dilation))
        return Conv2dWinograd_2_3;
      return Conv2dSlidingWindow;
        */
}

pub fn pack_weights_dw(
    context:        *mut Context,
    command_buffer: &mut CommandBuffer,
    pool:           &mut ResourcePool,
    weight:         &Tensor) -> VTensor {
    
    todo!();
        /*
            /* Source */
      const IntArrayRef src_filter = weight.sizes();
      const float* const src_weight_ptr = weight.data_ptr<float>();

      const i64 src_kw_sz = src_filter[Layout::Filter::width];
      const i64 src_kh_sz = src_filter[Layout::Filter::height];
      const i64 src_kernel_sz = src_kw_sz * src_kh_sz;
      const i64 src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];
      const i64 num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));

      /* Destination */
      const i64 dst_kw_sz = src_kernel_sz;
      const i64 dst_kh_sz = num_stacks;
      const i64 dst_kernel_sz = dst_kw_sz * dst_kh_sz;

      vTensor v_weight{
          context,
          &pool,
          {
              4,
              dst_kh_sz,
              dst_kw_sz,
          },
          weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_weight_payload = v_weight_future.wait();

      float* const dst_weight_ptr = v_weight_payload.get();
      memset(dst_weight_ptr, 0, v_weight.nbytes());

      for (i64 src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
        /* Source */
        const float* const src_weight_oc_ptr = src_weight_ptr + src_oc * src_block_sz;

        /* Destination */
        const i64 dst_oh = src_oc / 4;
        const i64 dst_c = src_oc % 4;

        float* const dst_weight_c_ptr = dst_weight_ptr +
                                        dst_c * dst_kernel_sz +
                                        dst_oh * dst_kw_sz;

        for (i64 src_ih = 0; src_ih < src_filter[Layout::Filter::height]; ++src_ih) {
          memcpy(
              dst_weight_c_ptr + src_ih * src_kw_sz,
              src_weight_oc_ptr + src_ih * src_kw_sz,
              sizeof(float) * src_kw_sz);
        }
      }

      return v_weight;
        */
}

pub fn pack_weights_2d(
    context:        *mut Context,
    command_buffer: &mut CommandBuffer,
    pool:           &mut ResourcePool,
    weight:         &Tensor) -> VTensor {
    
    todo!();
        /*
            /* Source */
      const IntArrayRef src_filter = weight.sizes();
      const float* const src_weight_ptr = weight.data_ptr<float>();

      const i64 src_kw_sz = src_filter[Layout::Filter::width];
      const i64 src_kh_sz = src_filter[Layout::Filter::height];
      const i64 src_kernel_sz = src_kw_sz * src_kh_sz;
      const i64 src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];

      const i64 num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));
      const i64 stack_depth = utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));

      /* Destination */
      const i64 dst_kw_sz = src_kw_sz * stack_depth;
      const i64 dst_kh_sz = src_kh_sz * num_stacks;
      const i64 dst_kernel_sz = dst_kw_sz * dst_kh_sz;

      vTensor v_weight{
          context,
          &pool,
          {
              4,
              dst_kh_sz,
              dst_kw_sz,
          },
          weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_weight_payload = v_weight_future.wait();

      float* const dst_weight_ptr = v_weight_payload.get();
      memset(dst_weight_ptr, 0, v_weight.nbytes());

      for (i64 src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
        /* Source */
        const float* const src_weight_oc_ptr = src_weight_ptr + src_oc * src_block_sz;

        /* Destination */
        const i64 dst_oh = src_oc / 4;
        const i64 dst_c = src_oc % 4;

        float* const dst_weight_c_ptr = dst_weight_ptr + dst_c * dst_kernel_sz;

        for (i64 src_ic = 0; src_ic < src_filter[Layout::Filter::input]; ++src_ic) {
          const i64 dst_ic4 = src_ic / 4;

          for (i64 src_ih = 0; src_ih < src_kh_sz; ++src_ih) {
            for (i64 src_iw = 0; src_iw < src_kw_sz; ++src_iw) {
              memcpy(
                  dst_weight_c_ptr + (dst_oh * src_kh_sz + src_ih) * dst_kw_sz +
                    dst_ic4 * src_kw_sz * 4 + src_iw * 4 + src_ic % 4,
                  src_weight_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
                  sizeof(float));
            }
          }
        }
      }

      return v_weight;
        */
}

pub fn pack_weights_2d_old(
        context:        *mut Context,
        command_buffer: &mut CommandBuffer,
        pool:           &mut ResourcePool,
        weight:         &Tensor) -> VTensor {
    
    todo!();
        /*
            const IntArrayRef src_filter = weight.sizes();
      const float* const src_weight_ptr = weight.data_ptr<float>();

      const u32 OC = src_filter[Layout::Filter::output];
      const u32 OC_4 = vulkan::utils::div_up(OC, 4u);
      const u32 C = src_filter[Layout::Filter::input];
      const u32 C_4 = vulkan::utils::div_up(C, 4u);
      const u32 KH = src_filter[Layout::Filter::height];
      const u32 KW = src_filter[Layout::Filter::width];

      vTensor v_weight{
        context,
        &pool,
        {
          1,
          4 * KH * KW,
          OC_4,
          4 * C_4
        },
        weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_weight_payload = v_weight_future.wait();

      float* const dst_weight_ptr = v_weight_payload.get();
      memset(dst_weight_ptr, 0, v_weight.nbytes());

      const float* const src = src_weight_ptr;
      float* const dst = dst_weight_ptr;

      {
        u32 ridx = 0;
        const u32 oc_4SizeNumel = KW * KH * C_4 * 16;
        for (u32 oc = 0; oc < OC; ++oc) {
          int oc_4 = oc / 4;
          int oc_4_i = oc % 4;
          float* dst_oc = dst + oc_4 * oc_4SizeNumel;
          for (u32 ic = 0; ic < C; ++ic) {
            int ic_4 = ic / 4;
            int ic_4_i = ic % 4;
            float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
            for (u32 ky = 0; ky < KH; ++ky) {
              float* dst_ky = dst_ic + ky * KW * 16;
              for (u32 kx = 0; kx < KW; ++kx) {
                float* dst_kx = dst_ky + kx * 16;
                dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
              }
            }
          }
        }

        // shader KO4C4HW_to_image
        struct Image3D {
          float* data_;
          u32 dim0_, dim1_, dim2_;

          Image3D(u32 dim0, u32 dim1, u32 dim2) {
            dim0_ = dim0;
            dim1_ = dim1;
            dim2_ = dim2;
            data_ = new float[dim0 * dim1 * dim2 * 4];  // TODO: memory leak
            memset(data_, 0.f, dim0 * dim1 * dim2 * 4 * sizeof(float));
          }

          inline u32 idx(u32 i0, u32 i1, u32 i2, u32 i3) {
            return i3 + i2 * 4 + i1 * 4 * dim2_ + i0 * 4 * dim2_ * dim1_;
          }

          void set(u32 i0, u32 i1, u32 i2, u32 i3, float value) {
            data_[idx(i0, i1, i2, i3)] = value;
          }

          float get(u32 i0, u32 i1, u32 i2, u32 i3) {
            return data_[idx(i0, i1, i2, i3)];
          }
        } image{4 * C_4, OC_4, KH * KW};

        for (u32 sx = 0; sx < C_4; ++sx) {
          for (u32 sy = 0; sy < OC_4; ++sy) {
            for (u32 sz = 0; sz < (KH * KW); ++sz) {
              for (u32 vi = 0; vi < 4; ++vi) {
                int bufferVIdx = 4 * sx * KH * KW + 4 * sy * C_4 * KH * KW + 4 * sz;
                image.set(4 * sx + 0, sy, sz, vi, dst[4 * (bufferVIdx + 0) + vi]);
                image.set(4 * sx + 1, sy, sz, vi, dst[4 * (bufferVIdx + 1) + vi]);
                image.set(4 * sx + 2, sy, sz, vi, dst[4 * (bufferVIdx + 2) + vi]);
                image.set(4 * sx + 3, sy, sz, vi, dst[4 * (bufferVIdx + 3) + vi]);
              }
            }
          }
        }

        // inverse function of nchw_to_image
        const u32 W = 4 * C_4;
        const u32 H = OC_4;
        const u32 D = KH * KW;
        for (u32 sx = 0; sx < W; ++sx) {
          for (u32 sy = 0; sy < H; ++sy) {
            for (u32 sz = 0; sz < D; ++sz) {
              for (u32 szvi = 0; szvi < 4; ++szvi) {
                dst_weight_ptr[W * sy + sx + (4 * sz + szvi) * W * H] = image.get(sx, sy, sz, szvi);
              }
            }
          }
        }
      }

      return v_weight;
        */
}

pub fn pack_weights_2d_winograd_2_3(
        context:        *mut Context,
        command_buffer: &mut CommandBuffer,
        pool:           &mut ResourcePool,
        weight:         &Tensor) -> VTensor {
    
    todo!();
        /*
            /* Source */
      const IntArrayRef src_filter = weight.sizes();

      TORCH_CHECK(
          src_filter[Layout::Filter::width] == 3 && src_filter[Layout::Filter::height] == 3,
          "Kernel size must be 3x3 for Winograd(2x2, 3x3)!");
      const i64 src_ic_sz = src_filter[Layout::Filter::input];
      const i64 src_oc_sz = src_filter[Layout::Filter::output];

      /* Destination */
      const i64 dst_ow_sz = div_up(src_ic_sz, INT64_C(4));
      const i64 dst_oh_sz = div_up(src_oc_sz, INT64_C(4));
      const i64 dst_kw_sz = 16*dst_ow_sz;
      const i64 dst_kh_sz = 4*dst_oh_sz;
      const i64 dst_block_sz = dst_kw_sz * dst_kh_sz;

      vTensor v_weight{
          context,
          &pool,
          {
            4,
            4*dst_oh_sz,
            16*dst_ow_sz,
          },
          weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_weight_payload = v_weight_future.wait();

      float* const dst_weight_ptr = v_weight_payload.get();
      memset(dst_weight_ptr, 0, v_weight.nbytes());

      for (i64 src_oc = 0; src_oc < src_oc_sz; ++src_oc) {
        const i64 dst_oh = src_oc / 4;
        const i64 dst_iw = src_oc % 4;

        for (i64 src_ic = 0; src_ic < src_ic_sz; ++src_ic) {
          const i64 dst_ow = src_ic / 4;
          const i64 dst_c = src_ic % 4;

          //const float* const src_k_ptr = src_weight_ptr + src_oc * src_block_sz + src_ic * 9;
          float* const dst_k = dst_weight_ptr + dst_c * dst_block_sz;

          const float s00 = weight[src_oc][src_ic][0][0].item<float>();
          const float s01 = weight[src_oc][src_ic][0][1].item<float>();
          const float s02 = weight[src_oc][src_ic][0][2].item<float>();
          const float s10 = weight[src_oc][src_ic][1][0].item<float>();
          const float s11 = weight[src_oc][src_ic][1][1].item<float>();
          const float s12 = weight[src_oc][src_ic][1][2].item<float>();
          const float s20 = weight[src_oc][src_ic][2][0].item<float>();
          const float s21 = weight[src_oc][src_ic][2][1].item<float>();
          const float s22 = weight[src_oc][src_ic][2][2].item<float>();

          const float m00 = s00;
          const float m01 = s01;
          const float m02 = s02;
          const float m10 = (s00 + s10 + s20)/2.f;
          const float m11 = (s01 + s11 + s21)/2.f;
          const float m12 = (s02 + s12 + s22)/2.f;
          const float m20 = (s00 - s10 + s20)/2.f;
          const float m21 = (s01 - s11 + s21)/2.f;
          const float m22 = (s02 - s12 + s22)/2.f;
          const float m30 = s20;
          const float m31 = s21;
          const float m32 = s22;

          dst_k[(4*dst_oh + 0)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m00;
          dst_k[(4*dst_oh + 0)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m00 + m01 + m02)/2.f;
          dst_k[(4*dst_oh + 0)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m00 - m01 + m02)/2.f;
          dst_k[(4*dst_oh + 0)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m02;
          dst_k[(4*dst_oh + 1)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m10;
          dst_k[(4*dst_oh + 1)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m10 + m11 + m12)/2.f;
          dst_k[(4*dst_oh + 1)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m10 - m11 + m12)/2.f;
          dst_k[(4*dst_oh + 1)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m12;
          dst_k[(4*dst_oh + 2)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m20;
          dst_k[(4*dst_oh + 2)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m20 + m21 + m22)/2.f;
          dst_k[(4*dst_oh + 2)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m20 - m21 + m22)/2.f;
          dst_k[(4*dst_oh + 2)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m22;
          dst_k[(4*dst_oh + 3)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m30;
          dst_k[(4*dst_oh + 3)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m30 + m31 + m32)/2.f;
          dst_k[(4*dst_oh + 3)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m30 - m31 + m32)/2.f;
          dst_k[(4*dst_oh + 3)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m32;
        }
      }

      return v_weight;
        */
}

pub fn pack_weights(
        pool:        &mut ResourcePool,
        weight_arg:  &Tensor,
        conv_method: Conv2dMethod) -> VTensor {
    
    todo!();
        /*
            if (weight_arg.is_vulkan()) {
        return convert(weight_arg);
      }

      Context* const context = context();
      Command::Buffer& command_buffer = context->command().pool.stream();

      const Tensor weight = weight_arg.contiguous();

      if (conv_method == Conv2dDepthwise) {
        return pack_weights_dw(
            context,
            command_buffer,
            pool,
            weight);
      }

      if (conv_method == Conv2dOld) {
        return pack_weights_2d_old(
            context,
            command_buffer,
            pool,
            weight);
      }

      if (conv_method == Conv2dWinograd_2_3) {
        return pack_weights_2d_winograd_2_3(
            context,
            command_buffer,
            pool,
            weight);
      }

      return pack_weights_2d(
          context,
          command_buffer,
          pool,
          weight);
        */
}

pub fn pack_biases(
        pool:   &mut ResourcePool,
        bias:   &Option<Tensor>,
        weight: &Tensor) -> VTensor {
    
    todo!();
        /*
            if (bias && bias->is_vulkan()) {
        return convert(*bias);
      }

      Context* const context = context();
      Command::Buffer& command_buffer = context->command().pool.stream();

      const i64 src_w = weight.size(Layout::Filter::output);
      const i64 packed_w = div_up(src_w, INT64_C(4));
      vTensor v_bias{
        context,
        &pool,
        {
          4,
          1,
          packed_w,
        },
        weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_bias_payload = v_bias_future.wait();

      if (bias) {
        const float* const src_bias_ptr = bias->contiguous().data_ptr<float>();
        float* const dst_bias_ptr = v_bias_payload.get();

        memset(dst_bias_ptr, 0, v_bias.nbytes());
        for (i64 i = 0; i < src_w; ++i) {
          const i64 c = i % 4;
          const i64 x = i / 4;
          dst_bias_ptr[c * packed_w + x] = src_bias_ptr[i];
        }
      }
      else {
        memset(
            v_bias_payload.get(),
            // 2's complement integers and IEEE-754 floating point numbers both
            // have identical bit representations for 0, so can use memset which
            // only accepts u8 parameter.
            0,
            v_bias.nbytes());
      }

      return v_bias;
        */
}

pub fn pack_filter(
        weight:   &Tensor,
        dilation: &[i32]) -> [i64; 4] {
    
    todo!();
        /*
            const IntArrayRef filter = weight.sizes();

      const auto effective = [](const i64 k, const i64 d) {
        return k + (k - 1) * (d - 1);
      };

      return {
        align_up(filter[Layout::Filter::output], INT64_C(4)),
        align_up(filter[Layout::Filter::input], INT64_C(4)),
        effective(
            filter[Layout::Filter::height],
            dilation[Layout::Parameter::height]),
        effective(
            filter[Layout::Filter::width],
            dilation[Layout::Parameter::width]),
      };
        */
}

pub fn pack_params(vector: &Vec<i64>) -> [i64; 2] {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(2u == vector.size(), "Invalid usage!");

      return {
        vector[0],
        vector[1],
      };
        */
}

pub fn available(
        weight:         &Tensor,
        bias:           &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        output_min:     &Option<Scalar>,
        output_max:     &Option<Scalar>) -> bool {
    
    todo!();
        /*
            return available() &&
             // Weight
             (4 == weight.ndimension()) &&
             (weight.size(Layout::Filter::height) > 0) &&
             (weight.size(Layout::Filter::width) > 0) &&
             ((weight.device().is_cpu()) ||
              (DeviceType_Vulkan == weight.device().type())) &&
             (kFloat == weight.scalar_type()) &&
             // Bias
             ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                           ((bias->device().is_cpu()) ||
                                            (DeviceType_Vulkan == bias->device().type())) &&
                                           (kFloat == bias->scalar_type()) &&
                                           (transposed ? false /* to be addded in the future */
                                                       : (weight.size(Layout::Filter::output) ==
                                                              bias->size(Layout::Filter::output))))
                                        : true) &&
             // Stride
             (stride[Layout::Parameter::height] > 0) &&
             (stride[Layout::Parameter::width] > 0) &&
             // Padding
             (padding[Layout::Parameter::height] >= 0) &&
             (padding[Layout::Parameter::width] >= 0) &&
             // Dilation
             (dilation[Layout::Parameter::height] > 0) &&
             (dilation[Layout::Parameter::width] > 0) &&
             // Groups
             (groups > 0) &&
             // Input
             (weight.size(Layout::Filter::input) > 0) &&
             // Output
             (weight.size(Layout::Filter::output) > 0) &&
             // Output - Groups
             ((weight.size(Layout::Filter::output) % groups) == 0) &&
             // Output Min / Max
             (!output_min || output_min->isFloatingPoint()) &&
             (!output_max || output_max->isFloatingPoint()) &&
             true;
        */
}


pub fn usable(input: &Tensor) -> bool {
    
    todo!();
        /*
            // Input
      return (4 == input.ndimension()) &&
             (DeviceType_Vulkan == input.device().type()) &&
             (kFloat == input.scalar_type()) &&
             (input.size(Layout::Activation4D::batch) >= 0) &&
             (input.size(Layout::Activation4D::channels) > 0) &&
             (input.size(Layout::Activation4D::height) > 0) &&
             (input.size(Layout::Activation4D::width) > 0) &&
             !input.requires_grad() &&
             true;
        */
}


pub fn conv2d_dw(
        context:    *mut Context,
        v_output:   &mut VTensor,
        v_input:    &VTensor,
        v_weight:   &VTensor,
        v_bias:     &VTensor,
        filter:     &[i32],
        src_filter: &[i32],
        stride:     &[i32],
        padding:    &[i32],
        dilation:   &[i32],
        output_min: f32,
        output_max: f32)  {
    
    todo!();
        /*
            bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
      TORCH_CHECK(valid, "Not Implemented!")

      Command::Pool& command_pool = context->command().pool;
      Command::Buffer& command_buffer = command_pool.stream();
      {
        const struct Block final {
          uvec3 extents;
          i32 src_filter_width;
          ivec4 kernel;
          ivec2 stride;
          ivec2 padding;
          ivec2 dilate;
          vec2 clamp;
        } block {
          v_output.extents(),
          safe_downcast<i32>(src_filter[Layout::Filter::width]),
          {
            safe_downcast<i32>(filter[Layout::Filter::width]),
            safe_downcast<i32>(filter[Layout::Filter::height]),
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::width]),
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::height]),
          },
          {
            safe_downcast<i32>(stride[Layout::Parameter::width]),
            safe_downcast<i32>(stride[Layout::Parameter::height]),
          },
          {
            safe_downcast<i32>(padding[Layout::Parameter::width]),
            safe_downcast<i32>(padding[Layout::Parameter::height]),
          },
          {
            safe_downcast<i32>(dilation[Layout::Parameter::width]),
            safe_downcast<i32>(dilation[Layout::Parameter::height]),
          },
          {
            output_min,
            output_max,
          },
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(conv2d_dw),
            v_output.extents(),
            context->gpu().adapter->local_work_group_size(),
            // Write-only access bypasses synchronization but inserts appropriate
            // barriers if necessary.
            v_output.image(
                command_buffer,
                vTensor::Stage::Compute,
                vTensor::Access::Write),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_input.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_weight.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_bias.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Object lifetime is managed by the resource pool.
            // It is OK not to keep track of the handle.
            context->resource().pool.uniform(block).object);
      }
      command_pool.submit(context->gpu().queue, command_buffer);
        */
}


pub fn conv2d_pw(
        context:    *mut Context,
        v_output:   &mut VTensor,
        v_input:    &VTensor,
        v_weight:   &VTensor,
        v_bias:     &VTensor,
        filter:     &[i32],
        src_filter: &[i32],
        stride:     &[i32],
        padding:    &[i32],
        dilation:   &[i32],
        output_min: f32,
        output_max: f32)  {
    
    todo!();
        /*
            bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
      TORCH_CHECK(valid, "Not Implemented!")

      Command::Pool& command_pool = context->command().pool;
      Command::Buffer& command_buffer = command_pool.stream();
      {
        const struct Block final {
          uvec3 extents;
          i32 ic;
          ivec2 stride;
          ivec2 padding;
          vec2 clamp;
        } block {
          v_output.extents(),
          safe_downcast<i32>(filter[Layout::Filter::input]),
          {
            safe_downcast<i32>(stride[Layout::Parameter::width]),
            safe_downcast<i32>(stride[Layout::Parameter::height]),
          },
          {
            safe_downcast<i32>(padding[Layout::Parameter::width]),
            safe_downcast<i32>(padding[Layout::Parameter::height]),
          },
          {
            output_min,
            output_max,
          },
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(conv2d_pw),
            v_output.extents(),
            context->gpu().adapter->local_work_group_size(),
            // Write-only access bypasses synchronization but inserts appropriate
            // barriers if necessary.
            v_output.image(
                command_buffer,
                vTensor::Stage::Compute,
                vTensor::Access::Write),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_input.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_weight.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_bias.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Object lifetime is managed by the resource pool.
            // It is OK not to keep track of the handle.
            context->resource().pool.uniform(block).object);
      }
      command_pool.submit(context->gpu().queue, command_buffer);
        */
}


pub fn conv2d(
        context:    *mut Context,
        v_output:   &mut VTensor,
        v_input:    &VTensor,
        v_weight:   &VTensor,
        v_bias:     &VTensor,
        filter:     &[i32],
        src_filter: &[i32],
        stride:     &[i32],
        padding:    &[i32],
        dilation:   &[i32],
        output_min: f32,
        output_max: f32)  {
    
    todo!();
        /*
            bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
      TORCH_CHECK(valid, "Not Implemented!")

      Command::Pool& command_pool = context->command().pool;
      Command::Buffer& command_buffer = command_pool.stream();
      {
        const struct Block final {
          uvec3 extents;
          i32 ic4;
          ivec4 kernel;
          ivec2 ikernel;
          ivec2 stride;
          ivec2 padding;
          ivec2 dilate;
          vec2 clamp;
          ivec4 src_filter;
        } block {
          v_output.extents(),
          safe_downcast<i32>(filter[Layout::Filter::input] / 4),
          {
            safe_downcast<i32>(filter[Layout::Filter::width]),
            safe_downcast<i32>(filter[Layout::Filter::height]),
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::width]),
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::height]),
          },
          {
            safe_downcast<i32>(src_filter[Layout::Filter::width] * 4),
            safe_downcast<i32>(src_filter[Layout::Filter::height]),
          },
          {
            safe_downcast<i32>(stride[Layout::Parameter::width]),
            safe_downcast<i32>(stride[Layout::Parameter::height]),
          },
          {
            safe_downcast<i32>(padding[Layout::Parameter::width]),
            safe_downcast<i32>(padding[Layout::Parameter::height]),
          },
          {
            safe_downcast<i32>(dilation[Layout::Parameter::width]),
            safe_downcast<i32>(dilation[Layout::Parameter::height]),
          },
          {
            output_min,
            output_max,
          },
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(conv2d),
            v_output.extents(),
            context->gpu().adapter->local_work_group_size(),
            // Write-only access bypasses synchronization but inserts appropriate
            // barriers if necessary.
            v_output.image(
                command_buffer,
                vTensor::Stage::Compute,
                vTensor::Access::Write),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_input.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_weight.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_bias.image(
                command_buffer,
                vTensor::Stage::Compute),
            // Object lifetime is managed by the resource pool.
            // It is OK not to keep track of the handle.
            context->resource().pool.uniform(block).object);
      }
      command_pool.submit(context->gpu().queue, command_buffer);
        */
}


pub fn conv2d_winograd_2_3(
        context:    *mut Context,
        v_output:   &mut VTensor,
        v_input:    &VTensor,
        v_weight:   &VTensor,
        v_bias:     &VTensor,
        filter:     &[i32],
        src_filter: &[i32],
        stride:     &[i32],
        padding:    &[i32],
        dilation:   &[i32],
        output_min: f32,
        output_max: f32)  {
    
    todo!();
        /*
            // Winograd(2x2, 3x3) calculates 2x2 tile of output for every subprogram
      const i64 out_h_units = div_up(v_output.sizes()[Layout::Activation4D::height], INT64_C(2));
      const i64 out_w_units = div_up(v_output.sizes()[Layout::Activation4D::width], INT64_C(2));

      bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
      TORCH_CHECK(valid, "Not Implemented!")

      Command::Pool& command_pool = context->command().pool;
      Command::Buffer& command_buffer = command_pool.stream();

      vTensor v_input_winograd{
        context,
        {
          v_input.sizes()[Layout::Activation4D::batch],
          v_input.sizes()[Layout::Activation4D::channels],
          out_h_units*4,
          out_w_units*4,
        },
        v_output.options(),
      };

      {
        const struct TransformBlock final {
          uvec3 extents;
          u32 fill;
          ivec2 limits;
          ivec2 padding;
        } transform_block {
          v_input_winograd.extents(),
          0u,
          {
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::width]),
            safe_downcast<i32>(v_input.sizes()[Layout::Activation4D::height]),
          },
          {
            safe_downcast<i32>(padding[Layout::Parameter::width]),
            safe_downcast<i32>(padding[Layout::Parameter::height]),
          },
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(transform_winograd_2_3_sh),
            v_input_winograd.extents(),
            context->gpu().adapter->local_work_group_size(),
            v_input_winograd.image(
                command_buffer,
                vTensor::Stage::Compute,
                vTensor::Access::Write),
            v_input.image(
                command_buffer,
                vTensor::Stage::Compute),
            context->resource().pool.uniform(transform_block).object);

      }
      {
        const struct Block final {
          uvec3 extents;
          i32 ic4;
          vec2 clamp;
        } block {
          v_output.extents(),
          safe_downcast<i32>(filter[Layout::Filter::input] / 4),
          {
            output_min,
            output_max,
          },
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(conv2d_winograd_2_3),
            {
              safe_downcast<u32>(out_w_units),
              safe_downcast<u32>(out_h_units),
              v_output.extents().data[2u],
            },
            context->gpu().adapter->local_work_group_size(),
            v_output.image(
                command_buffer,
                vTensor::Stage::Compute,
                vTensor::Access::Write),
            v_input_winograd.image(
                command_buffer,
                vTensor::Stage::Compute),
            v_weight.image(
                command_buffer,
                vTensor::Stage::Compute),
            v_bias.buffer(
                command_buffer,
                vTensor::Stage::Compute),
            context->resource().pool.uniform(block).object);
      }
      command_pool.submit(context->gpu().queue, command_buffer);
        */
}


pub fn conv2d_old(
        context:    *mut Context,
        v_output:   &mut VTensor,
        v_input:    &VTensor,
        v_weight:   &VTensor,
        v_bias:     &VTensor,
        filter:     &[i32],
        src_filter: &[i32],
        stride:     &[i32],
        padding:    &[i32],
        dilation:   &[i32],
        output_min: f32,
        output_max: f32)  {
    
    todo!();
        /*
            using namespace utils;
      bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
      TORCH_CHECK(valid, "Not Implemented!")

      Command::Pool& command_pool = context->command().pool;
      Command::Buffer& command_buffer = command_pool.stream();
      {
        const i32 W = v_input.extents().data[0];
        const i32 H = v_input.extents().data[1];
        const i32 C_4 = v_input.extents().data[2];
        const i32 C = 4 * C_4;

        const i32 OW = v_output.extents().data[0];
        const i32 OH = v_output.extents().data[1];
        const i32 OC_4 = v_output.extents().data[2];
        const i32 OC = 4 * OC_4;

        const struct Block final {
          i32 padding_x, padding_y;
          i32 kernel_x, kernel_y;
          i32 stride_x, stride_y;
          i32 dilate_x, dilate_y;
          i32 outputSize[4];
          i32 inputSize[4];
          float outputMin;
          float outputMax;
        } block {
          safe_downcast<i32>(padding[Layout::Parameter::width]),
          safe_downcast<i32>(padding[Layout::Parameter::height]),
          safe_downcast<i32>(filter[Layout::Filter::width]),
          safe_downcast<i32>(filter[Layout::Filter::height]),
          safe_downcast<i32>(stride[Layout::Parameter::width]),
          safe_downcast<i32>(stride[Layout::Parameter::height]),
          safe_downcast<i32>(dilation[Layout::Parameter::width]),
          safe_downcast<i32>(dilation[Layout::Parameter::height]),
          { OW, OH, OC_4, OC },
          { W, H, C_4, C },
          output_min,
          output_max,
        };

        context->dispatch(
            command_buffer,
            {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(conv2d_nogroup_clamp),
            v_output.extents(),
            context->gpu().adapter->local_work_group_size(),
            // Write-only access bypasses synchronization but inserts appropriate
            // barriers if necessary.
            v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_input.image(
              command_buffer,
              vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_weight.image(
              command_buffer,
              vTensor::Stage::Compute),
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_bias.buffer(
              command_buffer,
              vTensor::Stage::Compute),
            // Object lifetime is managed by the resource pool.
            // It is OK not to keep track of the handle.
            context->resource().pool.uniform(block).object);
      }
      command_pool.submit(context->gpu().queue, command_buffer);
        */
}

pub fn convolution(
        input:          &Tensor,
        weight:         &Tensor,
        bias:           &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64) -> Tensor {
    
    todo!();
        /*
            return Conv2dOpContext::create(
          context()->resource().pool,
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups
      ).run(input);
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("convolution_overrideable", convolution);
    }
    */
}

impl Conv2dOpContext {
    
    pub fn new(
        pool:           &mut ResourcePool,
        weight:         &Tensor,
        bias:           &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        method:         Conv2dMethod,
        output_min:     &Option<Scalar>,
        output_max:     &Option<Scalar>) -> Self {
    
        todo!();
        /*


            : packed_{
          pack_weights(pool, weight, method),
          pack_biases(pool, bias, weight),
          pack_filter(weight, expand_param_if_needed(dilation, "dilation", 2)),
          pack_params(expand_param_if_needed(stride, "stride", 2)),
          pack_params(expand_param_if_needed(padding, "padding", 2)),
          pack_params(expand_param_if_needed(dilation, "dilation", 2)),
          safe_downcast<i32>(groups),
          output_min ? output_min->template to<float>() : -numeric_limits<float>::infinity(),
          output_max ? output_max->template to<float>() : +numeric_limits<float>::infinity(),
        },
        unpacked_{
          weight,
          bias,
          weight.sizes().vec(),
          stride.vec(),
          padding.vec(),
          dilation.vec(),
          groups,
          output_min,
          output_max,
        },
        method_(method)
        */
    }
    
    pub fn create(&mut self, 
        pool:               &mut ResourcePool,
        weight:             &Tensor,
        bias:               &Option<Tensor>,
        stride_arg:         &[i32],
        padding_arg:        &[i32],
        dilation_arg:       &[i32],
        transposed:         bool,
        output_padding_arg: &[i32],
        groups:             i64,
        output_min:         &Option<Scalar>,
        output_max:         &Option<Scalar>) -> Conv2dOpContext {
        
        todo!();
        /*
            const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
      const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
      const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
      const auto output_padding = output_padding_arg; // TODO: Deconvolutions

      TORCH_CHECK(
          available(
              weight,
              bias,
              stride,
              padding,
              dilation,
              transposed,
              output_padding,
              groups,
              output_min,
              output_max),
          "Vulkan::convolution not available! "
          "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
          "transposed, output_padding, output_min, output_max) parameters are either "
          "invalid individually or their combination is not supported by Vulkan impl.");

      const auto method = determine_method(
          weight.sizes(),
          stride,
          padding,
          dilation,
          groups);

      // Pass in the originals
      return Conv2dOpContext{
        pool,
        weight,
        bias,
        stride_arg,
        padding_arg,
        dilation_arg,
        transposed,
        output_padding_arg,
        groups,
        method,
        output_min,
        output_max,
      };
        */
    }
    
    pub fn run(&self, input_arg: &Tensor) -> Tensor {
        
        todo!();
        /*
            Context* const context = context();

      const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
      const vTensor& v_input = convert(input);

      TORCH_CHECK(
          usable(input),
          "Vulkan Convolution not usable! "
          "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

      vTensor v_output{
        context,
        conv_output_size(
            v_input.sizes(),
            unpacked_.filter,
            packed_.padding,
            packed_.stride,
            packed_.dilation),
        input.options(),
      };

      {
        void (*conv_func) (
          Context* const,
          vTensor&,
          const vTensor&,
          const vTensor&,
          const vTensor&,
          const IntArrayRef,
          const IntArrayRef,
          const IntArrayRef,
          const IntArrayRef,
          const IntArrayRef,
          const float,
          const float
        );
        switch(method_) {
          case Conv2dDepthwise:
            conv_func = &conv2d_dw;
            break;
          case Conv2dPointwise:
            conv_func = &conv2d_pw;
            break;
          case Conv2dOld:
            conv_func = &conv2d_old;
            break;
          case Conv2dWinograd_2_3:
            conv_func = &conv2d_winograd_2_3;
            break;
          default:
            conv_func = &conv2d;
            break;
        }
        conv_func(
          context,
          v_output,
          v_input,
          packed_.v_weight,
          packed_.v_bias,
          packed_.filter,
          unpacked_.filter,
          packed_.stride,
          packed_.padding,
          packed_.dilation,
          packed_.output_min,
          packed_.output_max);
      }

      return convert(v_output);
        */
    }
    
    pub fn unpack(&self) -> Conv2dOpContextState {
        
        todo!();
        /*
            return Conv2dOpContext::State{
        unpacked_.weight,
        unpacked_.bias,
        unpacked_.stride,
        unpacked_.padding,
        unpacked_.dilation,
        unpacked_.groups,
        unpacked_.output_min,
        unpacked_.output_max,
      };
        */
    }
}

pub fn conv2d_clamp_prepack(
        weight:     Tensor,
        bias:       Option<Tensor>,
        stride:     Vec<i64>,
        padding:    Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
    
    todo!();
        /*
            return make_intrusive<Conv2dOpContext>(
          Conv2dOpContext::create(
              persistent()->pool,
              move(weight),
              move(bias),
              move(stride),
              move(padding),
              move(dilation),
              /* transposed = */ false,
              /* output_padding = */ {},
              groups,
              output_min,
              output_max));
        */
}

pub fn conv2d_clamp_run(
        input:   &Tensor,
        context: &IntrusivePtr<Conv2dOpContext>) -> Tensor {
    
    todo!();
        /*
            return context->run(input);
        */
}
