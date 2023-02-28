crate::ix!();

#[inline] pub fn is_pointer_aligned<T>(p: *mut T, align: libc::size_t) -> bool {
    todo!();
    /*
        return (reinterpret_cast<uintptr_t>(p) % align == 0);
    */
}

#[cfg(target_feature = "neon")]
#[inline] pub fn vert_sum_f32(
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t) -> float32x4_t 
{
    todo!();
    /*
        v0 = vaddq_f32(v0, v1);
      v2 = vaddq_f32(v2, v3);
      return vaddq_f32(v0, v2);
    */
}

#[cfg(target_feature = "neon")]
#[inline] pub fn horizontal_sum_f32(
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t) -> f32 
{
    todo!();
    /*
        v0 = vert_sum_f32(v0, v1, v2, v3);
      float32x2_t v = vadd_f32(vget_high_f32(v0), vget_low_f32(v0));
      return vget_lane_f32(vpadd_f32(v, v), 0);
    */
}


// Load/store functions that assume alignment
#[cfg(target_feature = "neon")]
#[inline] pub fn vld1q_f32_aligned(p: *const f32) -> float32x4_t {
    
    todo!();
    /*
        return vld1q_f32((const float*)
                       __builtin_assume_aligned(p, sizeof(float32x4_t)));
    */
}

#[cfg(target_feature = "neon")]
#[inline] pub fn vst1q_f32_aligned(p: *mut f32, v: float32x4_t)  {
    
    todo!();
    /*
        vst1q_f32((float*) __builtin_assume_aligned(p, sizeof(float32x4_t)), v);
    */
}

#[cfg(target_feature = "neon")]
#[inline] pub fn vst4_u8_aligned(p: *mut u8, v: uint8x8x4_t)  {
    
    todo!();
    /*
        vst4_u8((uint8_t*)
              __builtin_assume_aligned(p, sizeof(uint8x8x4_t)), v);
    */
}
