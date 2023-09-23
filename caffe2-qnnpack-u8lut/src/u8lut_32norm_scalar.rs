crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8lut32norm/scalar.c]

#[inline] pub fn compute_sum(
        n: usize,
        x: *const u8,
        t: *const u32) -> u32 {
    
    todo!();
        /*
            assert(n != 0);

      u32 vsum = 0;
      do {
        const usize vx = *x++;
        vsum += t[vx];
      } while (--n != 0);
      return vsum;
        */
}

pub fn pytorch_u8lut_32norm_ukernel_scalar(
        n: usize,
        x: *const u8,
        t: *const u32,
        y: *mut u8)  {
    
    todo!();
        /*
            assert(n != 0);

      const u32 vsum = compute_sum(n, x, t);
      assert(vsum != 0);

      struct fxdiv_divisor_u32 vsum_divisor = fxdiv_init_u32(vsum);
      const u32 vrounding = (vsum >> 1);
      do {
        const usize vx = *x++;
        const u32 vt = t[vx];
        const u32 vq =
            fxdiv_quotient_u32((vt << 8) + vrounding, vsum_divisor);
        const u8 vy = vq > 255 ? UINT8_C(255) : (u8)vq;
        *y++ = vy;
      } while (--n != 0);
        */
}
