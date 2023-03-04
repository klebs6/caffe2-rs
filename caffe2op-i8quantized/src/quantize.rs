crate::ix!();

pub struct Int8QuantizeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl Int8QuantizeOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
        Y->t.ResizeLike(X);
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;
        Int8Quantize(
            X.data<float>(),
            Y->t.mutable_data<uint8_t>(),
            X.numel(),
            Y_scale,
            Y_offset);
        return true;
        */
    }
}

register_cpu_operator!{Int8Quantize, int8::Int8QuantizeOp}

num_inputs!{Int8Quantize, (1,2)}

num_outputs!{Int8Quantize, 1}

inputs!{Int8Quantize, 
    0 => ("X", "FP32 Tensor X."),
    1 => ("Qparam", "Optional Qparam blob that contains quant param computed 
        on activation histogram data Will overwrite Y_scale and Y_zero_point argument if specified")
}

outputs!{Int8Quantize, 
    0 => ("Y", "Int8 Tensor qX representing X with linear quantization.")
}

args!{Int8Quantize, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset")
}

tensor_inference_function!{Int8Quantize, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      out.emplace_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}


#[inline] pub fn int_8quantize(
    input:     *const f32,
    out:       *mut u8,
    n:         i64,
    y_scale:   f32,
    y_offset:  i32)  
{
    todo!();
    /*
        const float inv_scale = 1.0f / Y_scale;
      uint32_t i = 0;

    #ifdef INT8_NEON_SIMD
      const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
      // magic float and magic int to take care of rounding
      // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
      // Some detail:
      // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
      // add a small number to a large number, the result rounds to the precision of
      // the least significant bit of the large number. For IEEE-754
      // single-precision number mantissa has 23 bits, and adding 2**23 would cause
      // rounding to the nearest even integer. The we cast to int and subtract the
      // same number (0x4B400000 is the integer representation of 12582912.0f) to
      // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
      // sign for negative numbers.
      const int32x4_t voffset = vdupq_n_s32(Y_offset - 0x4B400000);
      const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
      for (i = 0; i + 8 < N; i += 8) {
        const float32x4_t vin0123 = vld1q_f32(in);
        in += 4;
        const float32x4_t vin4567 = vld1q_f32(in);
        in += 4;
        const int32x4_t vraw0123 = vaddq_s32(
            voffset,
            vreinterpretq_s32_f32(
                vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
        const int32x4_t vraw4567 = vaddq_s32(
            voffset,
            vreinterpretq_s32_f32(
                vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
        const int16x8_t vraw01234567 =
            vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
        const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
        vst1_u8(out, vout01234567);
        out += 8;
      }
    #endif
      for (; i < N; ++i) {
        (*out++) = QuantizeUint8(Y_scale, Y_offset, (*in++));
      }
    */
}
