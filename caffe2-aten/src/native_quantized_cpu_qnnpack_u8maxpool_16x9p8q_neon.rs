// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8maxpool/16x9p8q-neon.c]

pub fn pytorch_u8maxpool_ukernel_16x9p8q_neon(
        n:                usize,
        ks:               usize,
        kc:               usize,
        input:            *const *const u8,
        output:           *mut u8,
        input_increment:  usize,
        output_increment: usize,
        params:           [PyTorchQnnpU8ClampingParams; 1])  {
    
    todo!();
        /*
            assert(n != 0);
      assert(ks != 0);
      assert(kc >= 16);

      const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);
      const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
      do {
        u8* o = output;
        {
          const u8* i0 = *input++;
          const u8* i1 = *input++;
          const u8* i2 = *input++;
          const u8* i3 = *input++;
          const u8* i4 = *input++;
          const u8* i5 = *input++;
          const u8* i6 = *input++;
          const u8* i7 = *input++;
          const u8* i8 = *input++;
          if (ks < 2) {
            i1 = i0;
          }
          if (ks <= 2) {
            i2 = i0;
          }
          if (ks < 4) {
            i3 = i0;
          }
          if (ks <= 4) {
            i4 = i0;
          }
          if (ks < 6) {
            i5 = i0;
          }
          if (ks <= 6) {
            i6 = i0;
          }
          if (ks < 8) {
            i7 = i0;
          }
          if (ks <= 8) {
            i8 = i0;
          }

          usize k = kc;
          while (k >= 16) {
            const uint8x16_t vi0 = vld1q_u8(i0);
            i0 += 16;
            const uint8x16_t vi1 = vld1q_u8(i1);
            i1 += 16;
            const uint8x16_t vi2 = vld1q_u8(i2);
            i2 += 16;
            const uint8x16_t vi3 = vld1q_u8(i3);
            i3 += 16;
            const uint8x16_t vi4 = vld1q_u8(i4);
            i4 += 16;
            const uint8x16_t vi5 = vld1q_u8(i5);
            i5 += 16;
            const uint8x16_t vi6 = vld1q_u8(i6);
            i6 += 16;
            const uint8x16_t vi7 = vld1q_u8(i7);
            i7 += 16;
            const uint8x16_t vi8 = vld1q_u8(i8);
            i8 += 16;

            const uint8x16_t vmax018 = vmaxq_u8(vmaxq_u8(vi0, vi1), vi8);
            const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
            const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
            const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

            const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
            const uint8x16_t vmax01678 = vmaxq_u8(vmax018, vmax67);
            const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax01678);
            const uint8x16_t vout =
                vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

            vst1q_u8(o, vout);
            o += 16;

            k -= 16;
          }
          if (k != 0) {
            const usize address_increment = k - 16;
            i0 = (const u8*)((uintptr_t)i0 + address_increment);
            i1 = (const u8*)((uintptr_t)i1 + address_increment);
            i2 = (const u8*)((uintptr_t)i2 + address_increment);
            i3 = (const u8*)((uintptr_t)i3 + address_increment);
            i4 = (const u8*)((uintptr_t)i4 + address_increment);
            i5 = (const u8*)((uintptr_t)i5 + address_increment);
            i6 = (const u8*)((uintptr_t)i6 + address_increment);
            i7 = (const u8*)((uintptr_t)i7 + address_increment);
            i8 = (const u8*)((uintptr_t)i8 + address_increment);
            o = (u8*)((uintptr_t)o + address_increment);

            const uint8x16_t vi0 = vld1q_u8(i0);
            const uint8x16_t vi1 = vld1q_u8(i1);
            const uint8x16_t vi2 = vld1q_u8(i2);
            const uint8x16_t vi3 = vld1q_u8(i3);
            const uint8x16_t vi4 = vld1q_u8(i4);
            const uint8x16_t vi5 = vld1q_u8(i5);
            const uint8x16_t vi6 = vld1q_u8(i6);
            const uint8x16_t vi7 = vld1q_u8(i7);
            const uint8x16_t vi8 = vld1q_u8(i8);

            const uint8x16_t vmax018 = vmaxq_u8(vmaxq_u8(vi0, vi1), vi8);
            const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
            const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
            const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

            const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
            const uint8x16_t vmax01678 = vmaxq_u8(vmax018, vmax67);
            const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax01678);
            const uint8x16_t vout =
                vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

            vst1q_u8(o, vout);
            o += 16;
          }
        }

        for (ptrdiff_t m = (ptrdiff_t)ks - 9; m > 0; m -= 8) {
          const u8* i0 = *input++;
          const u8* i1 = *input++;
          const u8* i2 = *input++;
          const u8* i3 = *input++;
          const u8* i4 = *input++;
          const u8* i5 = *input++;
          const u8* i6 = *input++;
          const u8* i7 = *input++;
          if (m < 2) {
            i1 = i0;
          }
          if (m <= 2) {
            i2 = i0;
          }
          if (m < 4) {
            i3 = i0;
          }
          if (m <= 4) {
            i4 = i0;
          }
          if (m < 6) {
            i5 = i0;
          }
          if (m <= 6) {
            i6 = i0;
          }
          if (m < 8) {
            i7 = i0;
          }

          o = output;
          usize k = kc;
          while (k >= 16) {
            const uint8x16_t vi0 = vld1q_u8(i0);
            i0 += 16;
            const uint8x16_t vi1 = vld1q_u8(i1);
            i1 += 16;
            const uint8x16_t vi2 = vld1q_u8(i2);
            i2 += 16;
            const uint8x16_t vi3 = vld1q_u8(i3);
            i3 += 16;
            const uint8x16_t vi4 = vld1q_u8(i4);
            i4 += 16;
            const uint8x16_t vi5 = vld1q_u8(i5);
            i5 += 16;
            const uint8x16_t vi6 = vld1q_u8(i6);
            i6 += 16;
            const uint8x16_t vi7 = vld1q_u8(i7);
            i7 += 16;
            const uint8x16_t vo = vld1q_u8(o);

            const uint8x16_t vmax01 = vmaxq_u8(vmaxq_u8(vi0, vi1), vo);
            const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
            const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
            const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

            const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
            const uint8x16_t vmax0167 = vmaxq_u8(vmax01, vmax67);
            const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax0167);
            const uint8x16_t vout =
                vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

            vst1q_u8(o, vout);
            o += 16;

            k -= 16;
          }
          if (k != 0) {
            const usize address_increment = k - 16;
            i0 = (const u8*)((uintptr_t)i0 + address_increment);
            i1 = (const u8*)((uintptr_t)i1 + address_increment);
            i2 = (const u8*)((uintptr_t)i2 + address_increment);
            i3 = (const u8*)((uintptr_t)i3 + address_increment);
            i4 = (const u8*)((uintptr_t)i4 + address_increment);
            i5 = (const u8*)((uintptr_t)i5 + address_increment);
            i6 = (const u8*)((uintptr_t)i6 + address_increment);
            i7 = (const u8*)((uintptr_t)i7 + address_increment);
            o = (u8*)((uintptr_t)o + address_increment);

            const uint8x16_t vi0 = vld1q_u8(i0);
            const uint8x16_t vi1 = vld1q_u8(i1);
            const uint8x16_t vi2 = vld1q_u8(i2);
            const uint8x16_t vi3 = vld1q_u8(i3);
            const uint8x16_t vi4 = vld1q_u8(i4);
            const uint8x16_t vi5 = vld1q_u8(i5);
            const uint8x16_t vi6 = vld1q_u8(i6);
            const uint8x16_t vi7 = vld1q_u8(i7);
            const uint8x16_t vo = vld1q_u8(o);

            const uint8x16_t vmax01 = vmaxq_u8(vmaxq_u8(vi0, vi1), vo);
            const uint8x16_t vmax23 = vmaxq_u8(vi2, vi3);
            const uint8x16_t vmax45 = vmaxq_u8(vi4, vi5);
            const uint8x16_t vmax67 = vmaxq_u8(vi6, vi7);

            const uint8x16_t vmax2345 = vmaxq_u8(vmax23, vmax45);
            const uint8x16_t vmax0167 = vmaxq_u8(vmax01, vmax67);
            const uint8x16_t vmax = vmaxq_u8(vmax2345, vmax0167);
            const uint8x16_t vout =
                vmaxq_u8(vminq_u8(vmax, voutput_max), voutput_min);

            vst1q_u8(o, vout);
            o += 16;
          }
        }
        input = (const u8**)((uintptr_t)input + input_increment);
        output = (u8*)((uintptr_t)o + output_increment);
      } while (--n != 0);
        */
}
