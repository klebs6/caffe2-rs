crate::ix!();

#[inline] pub fn clamped_cast<T>(f: f32) -> T {

    todo!();
    /*
        if (f >= T::max) {
        return T::max;
      }
      if (f <= T::min) {
        return T::min;
      }
      return static_cast<T>(f);
    */
}

#[cfg(arm_neon)]
#[inline] pub fn to_v4_f32(v: uint16x4_t) -> float32x4_t {
    
    todo!();
    /*
        return vcvtq_f32_u32(vmovl_u16(v));
    */
}

#[cfg(arm_neon)]
#[inline] pub fn to_f32_v4_x4(v: uint8x16_t) -> float32x4x4_t {
    
    todo!();
    /*
        float32x4x4_t out;

      uint16x8_t lo_u16 = vmovl_u8(vget_low_u8(v));

      out.val[0] = to_v4_f32(vget_low_u16(lo_u16));
      out.val[1] = to_v4_f32(vget_high_u16(lo_u16));

      uint16x8_t hi_u16 = vmovl_u8(vget_high_u8(v));

      out.val[2] = to_v4_f32(vget_low_u16(hi_u16));
      out.val[3] = to_v4_f32(vget_high_u16(hi_u16));

      return out;
    */
}

#[cfg(arm_neon)]
#[inline] pub fn clamp(v: &mut float32x4_t)  {
    
    todo!();
    /*
        v = vmaxq_f32(v, vdupq_n_f32(0));
      v = vminq_f32(v, vdupq_n_f32((float)uint8_t::max));
    */
}

#[cfg(arm_neon)]
#[inline] pub fn add_mean_and_clamp(v: &mut float32x4_t, mean: f32)  {
    
    todo!();
    /*
        v = vaddq_f32(v, vdupq_n_f32(mean));
      clamp(v);
    */
}

#[cfg(arm_neon)]
#[inline] pub fn convert_narrow_and_pack(v0: float32x4_t, v1: float32x4_t) -> uint8x8_t {
    
    todo!();
    /*
        uint16x4_t u16_0 = vmovn_u32(vcvtq_u32_f32(v0));
      uint16x4_t u16_1 = vmovn_u32(vcvtq_u32_f32(v1));
      uint16x8_t u16_01 = vcombine_u16(u16_0, u16_1);
      return vmovn_u16(u16_01);
    */
}
