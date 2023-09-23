crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8rmax/neon.c]

pub fn pytorch_u8rmax_ukernel_neon(
        n: usize,
        x: *const u8) -> u8 {
    
    todo!();
        /*
            assert(n != 0);

      if
        PYTORCH_QNNP_LIKELY(n >= 16) {
          uint8x16_t vmax = vmovq_n_u8(0);
          do {
            const uint8x16_t vx = vld1q_u8(x);
            x += 16;
            vmax = vmaxq_u8(vmax, vx);
            n -= 16;
          } while (n >= 16);
          if (n != 0) {
            const usize x_increment = n - 16;
            x = (const u8*)((uintptr_t)x + x_increment);
            const uint8x16_t vx = vld1q_u8(x);
            vmax = vmaxq_u8(vmax, vx);
          }
          uint8x8_t vmax8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));
          const uint8x8_t vmax4 = vpmax_u8(vmax8, vmax8);
          const uint8x8_t vmax2 = vpmax_u8(vmax4, vmax4);
          const uint8x8_t vmax1 = vpmax_u8(vmax2, vmax2);
          return vget_lane_u8(vmax1, 0);
        }
      else {
        uint8x8_t vmax = vmov_n_u8(0);
        do {
          const uint8x8_t vx = vld1_dup_u8(x);
          x += 1;
          vmax = vmax_u8(vmax, vx);
        } while (--n != 0);
        return vget_lane_u8(vmax, 0);
      }
        */
}
