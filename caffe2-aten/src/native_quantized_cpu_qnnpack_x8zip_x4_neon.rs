// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/x4-neon.c]

pub fn pytorch_qnnp_x8zip_x4_neon(
    n:      usize,
    input:  *const c_void,
    output: *mut c_void)  {
    
    todo!();
        /*
            const u8* x = input;
      const u8* y = x + n;
      const u8* z = y + n;
      const u8* w = z + n;
      u8* o = output;

      if (n >= 8) {
        do {
          uint8x8x4_t vxyzw;
          vxyzw.val[0] = vld1_u8(x);
          x += 8;
          vxyzw.val[1] = vld1_u8(y);
          y += 8;
          vxyzw.val[2] = vld1_u8(z);
          z += 8;
          vxyzw.val[3] = vld1_u8(w);
          w += 8;
          vst4_u8(o, vxyzw);
          o += 32;
          n -= 8;
        } while (n >= 8);
        if (n != 0) {
          const usize address_increment = n - 8;
          uint8x8x4_t vxyzw;
          vxyzw.val[0] = vld1_u8(x + address_increment);
          vxyzw.val[1] = vld1_u8(y + address_increment);
          vxyzw.val[2] = vld1_u8(z + address_increment);
          vxyzw.val[3] = vld1_u8(w + address_increment);
          vst4_u8((u8*)((uintptr_t)o + address_increment * 4), vxyzw);
        }
      } else {
        do {
          const u8 vx = *x++;
          const u8 vy = *y++;
          const u8 vz = *z++;
          const u8 vw = *w++;
          o[0] = vx;
          o[1] = vy;
          o[2] = vz;
          o[3] = vw;
          o += 4;
        } while (--n != 0);
      }
        */
}


