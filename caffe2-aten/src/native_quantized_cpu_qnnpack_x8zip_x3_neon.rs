//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/x3-neon.c]


pub fn pytorch_qnnp_x8zip_x3_neon(
        n:      usize,
        input:  *const void,
        output: *mut void)  {
    
    todo!();
        /*
            const u8* x = input;
      const u8* y = x + n;
      const u8* z = y + n;
      u8* o = output;

      if (n >= 8) {
        do {
          uint8x8x3_t vxyz;
          vxyz.val[0] = vld1_u8(x);
          x += 8;
          vxyz.val[1] = vld1_u8(y);
          y += 8;
          vxyz.val[2] = vld1_u8(z);
          z += 8;
          vst3_u8(o, vxyz);
          o += 24;
          n -= 8;
        } while (n >= 8);
        if (n != 0) {
          const usize address_increment = n - 8;
          uint8x8x3_t vxyz;
          vxyz.val[0] = vld1_u8(x + address_increment);
          vxyz.val[1] = vld1_u8(y + address_increment);
          vxyz.val[2] = vld1_u8(z + address_increment);
          vst3_u8((u8*)((uintptr_t)o + address_increment * 3), vxyz);
        }
      } else {
        do {
          const u8 vx = *x++;
          const u8 vy = *y++;
          const u8 vz = *z++;
          o[0] = vx;
          o[1] = vy;
          o[2] = vz;
          o += 3;
        } while (--n != 0);
      }
        */
}


