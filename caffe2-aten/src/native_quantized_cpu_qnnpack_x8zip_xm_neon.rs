// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-neon.c]

pub fn pytorch_qnnp_x8zip_xm_neon(
        n:      usize,
        m:      usize,
        input:  *const void,
        output: *mut void)  {
    
    todo!();
        /*
            const u8* w = input;
      const usize input_increment = n * 3;
      const usize output_increment = 4 - m * n;
      const u8* last_input = w + n * (m - 1);
      void* last_output = (void*)((uintptr_t)output + (m - 4));

      if (n >= 8) {
        for (usize i = 0; i < m; i += 4) {
          usize k = n;
          w = (const u8*)((uintptr_t)w + input_increment);
          if (w >= last_input) {
            w = last_input;
          }
          const u8* z = (const u8*)((uintptr_t)w - n);
          const u8* y = (const u8*)((uintptr_t)z - n);
          const u8* x = (const u8*)((uintptr_t)y - n);
          while (k >= 8) {
            const uint8x8_t vx = vld1_u8(x);
            x += 8;
            const uint8x8_t vy = vld1_u8(y);
            y += 8;
            const uint8x8_t vz = vld1_u8(z);
            z += 8;
            const uint8x8_t vw = vld1_u8(w);
            w += 8;

            const uint8x8x2_t vxy = vzip_u8(vx, vy);
            const uint8x8x2_t vzw = vzip_u8(vz, vw);
            const uint16x4x2_t vxyzw_lo = vzip_u16(
                vreinterpret_u16_u8(vxy.val[0]), vreinterpret_u16_u8(vzw.val[0]));
            const uint16x4x2_t vxyzw_hi = vzip_u16(
                vreinterpret_u16_u8(vxy.val[1]), vreinterpret_u16_u8(vzw.val[1]));

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_lo.val[0]),
                0);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_lo.val[0]),
                1);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_lo.val[1]),
                0);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_lo.val[1]),
                1);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_hi.val[0]),
                0);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_hi.val[0]),
                1);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_hi.val[1]),
                0);
            output = (void*)((uintptr_t)output + m);

            vst1_lane_u32(
                __builtin_assume_aligned(output, 1),
                vreinterpret_u32_u16(vxyzw_hi.val[1]),
                1);
            output = (void*)((uintptr_t)output + m);

            k -= 8;
          }
          if (k != 0) {
            const usize address_increment = k - 8;
            x = (const u8*)((uintptr_t)x + address_increment);
            y = (const u8*)((uintptr_t)y + address_increment);
            z = (const u8*)((uintptr_t)z + address_increment);
            w = (const u8*)((uintptr_t)w + address_increment);
            const int64x1_t vshift = vmov_n_s64(8 * address_increment);

            const uint64x1_t vx = vshl_u64(vreinterpret_u64_u8(vld1_u8(x)), vshift);
            const uint64x1_t vy = vshl_u64(vreinterpret_u64_u8(vld1_u8(y)), vshift);
            const uint64x1_t vz = vshl_u64(vreinterpret_u64_u8(vld1_u8(z)), vshift);
            const uint64x1_t vw = vshl_u64(vreinterpret_u64_u8(vld1_u8(w)), vshift);
            w += 8;
            const uint8x8x2_t vxy =
                vzip_u8(vreinterpret_u8_u64(vx), vreinterpret_u8_u64(vy));
            const uint8x8x2_t vzw =
                vzip_u8(vreinterpret_u8_u64(vz), vreinterpret_u8_u64(vw));
            const uint16x4x2_t vxyzw_lo = vzip_u16(
                vreinterpret_u16_u8(vxy.val[0]), vreinterpret_u16_u8(vzw.val[0]));
            const uint16x4x2_t vxyzw_hi = vzip_u16(
                vreinterpret_u16_u8(vxy.val[1]), vreinterpret_u16_u8(vzw.val[1]));

            uint32x2_t vxyzw0 = vreinterpret_u32_u16(vxyzw_lo.val[0]);
            uint32x2_t vxyzw1 = vreinterpret_u32_u16(vxyzw_lo.val[1]);
            uint32x2_t vxyzw2 = vreinterpret_u32_u16(vxyzw_hi.val[0]);
            uint32x2_t vxyzw3 = vreinterpret_u32_u16(vxyzw_hi.val[1]);

            if (k & 4) {
              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
              output = (void*)((uintptr_t)output + m);

              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 1);
              output = (void*)((uintptr_t)output + m);

              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw1, 0);
              output = (void*)((uintptr_t)output + m);

              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw1, 1);
              output = (void*)((uintptr_t)output + m);

              vxyzw0 = vxyzw2;
              vxyzw1 = vxyzw3;
            }

            if (k & 2) {
              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
              output = (void*)((uintptr_t)output + m);

              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 1);
              output = (void*)((uintptr_t)output + m);

              vxyzw0 = vxyzw1;
            }
            if (k & 1) {
              vst1_lane_u32(__builtin_assume_aligned(output, 1), vxyzw0, 0);
              output = (void*)((uintptr_t)output + m);
            }
          }
          output = (void*)((uintptr_t)output + output_increment);
          if (output > last_output) {
            output = last_output;
          }
        }
      } else {
        const u8* i = input;
        u8* o = output;
        usize k = n;
        do {
          usize l = m;
          const u8* ii = i++;
          do {
            *o++ = *ii;
            ii += n;
          } while (--l != 0);
        } while (--k != 0);
      }
        */
}

