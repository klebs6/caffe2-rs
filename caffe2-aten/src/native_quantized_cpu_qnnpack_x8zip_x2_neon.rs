//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/x2-neon.c]


pub fn pytorch_qnnp_x8zip_x2_neon(
        n:      usize,
        input:  *const void,
        output: *mut void)  {
    
    todo!();
        /*
            const u8* x = input;
      const u8* y = x + n;
      u8* o = output;

      if (n >= 8) {
        do {
          uint8x8x2_t vxy;
          vxy.val[0] = vld1_u8(x);
          x += 8;
          vxy.val[1] = vld1_u8(y);
          y += 8;
          vst2_u8(o, vxy);
          o += 16;
          ;
          n -= 8;
        } while (n >= 8);
        if (n != 0) {
          const usize address_increment = n - 8;
          uint8x8x2_t vxy;
          vxy.val[0] = vld1_u8((const u8*)((uintptr_t)x + address_increment));
          vxy.val[1] = vld1_u8((const u8*)((uintptr_t)y + address_increment));
          vst2_u8((u8*)((uintptr_t)o + address_increment * 2), vxy);
        }
      } else {
        do {
          const u8 vx = *x++;
          const u8 vy = *y++;
          o[0] = vx;
          o[1] = vy;
          o += 2;
        } while (--n != 0);
      }
        */
}


