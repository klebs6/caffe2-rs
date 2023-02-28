// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8clamp/neon.c]

pub fn pytorch_u8clamp_ukernel_neon(
    n:      usize,
    x:      *const u8,
    y:      *mut u8,
    params: [PyTorchQnnpU8ClampingParams; 1])  {
    
    todo!();
        /*
            assert(n != 0);

      const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
      const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);

      if
        PYTORCH_QNNP_LIKELY(n >= 8) {
          for (; n >= 64; n -= 64) {
            const uint8x16_t vx0 = vld1q_u8(x);
            x += 16;
            const uint8x16_t vx1 = vld1q_u8(x);
            x += 16;
            const uint8x16_t vx2 = vld1q_u8(x);
            x += 16;
            const uint8x16_t vx3 = vld1q_u8(x);
            x += 16;

            const uint8x16_t vy0 =
                vminq_u8(vmaxq_u8(vx0, voutput_min), voutput_max);
            const uint8x16_t vy1 =
                vminq_u8(vmaxq_u8(vx1, voutput_min), voutput_max);
            const uint8x16_t vy2 =
                vminq_u8(vmaxq_u8(vx2, voutput_min), voutput_max);
            const uint8x16_t vy3 =
                vminq_u8(vmaxq_u8(vx3, voutput_min), voutput_max);

            __builtin_prefetch(x + 640);

            vst1q_u8(y, vy0);
            y += 16;
            vst1q_u8(y, vy1);
            y += 16;
            vst1q_u8(y, vy2);
            y += 16;
            vst1q_u8(y, vy3);
            y += 16;
          }
          for (; n >= 8; n -= 8) {
            uint8x8_t vout = vld1_u8(x);
            x += 8;
            vout = vmin_u8(vout, vget_low_u8(voutput_max));
            vout = vmax_u8(vout, vget_low_u8(voutput_min));
            vst1_u8(y, vout);
            y += 8;
          }
          if (n != 0) {
            const usize n_increment = n - 8;
            x = (const u8*)((uintptr_t)x + n_increment);
            y = (u8*)((uintptr_t)y + n_increment);

            uint8x8_t vout = vld1_u8(x);
            vout = vmin_u8(vout, vget_low_u8(voutput_max));
            vout = vmax_u8(vout, vget_low_u8(voutput_min));
            vst1_u8(y, vout);
          }
        }
      else {
        do {
          uint8x8_t vout = vld1_dup_u8(x);
          x += 1;
          vout = vmin_u8(vout, vget_low_u8(voutput_max));
          vout = vmax_u8(vout, vget_low_u8(voutput_min));
          vst1_lane_u8(y, vout, 0);
          y += 1;
        } while (--n != 0);
      }
        */
}
