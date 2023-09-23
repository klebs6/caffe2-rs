/*!
  | Workaround for missing vld1_*_x2 and
  | vst1_*_x2 intrinsics in gcc-7.
  |
  */

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/missing_vld1_neon.h]

#[inline(always)] pub fn vld1_u8_x2(a: *const u8) -> uint8x8x2_t {
    
    todo!();
        /*
            uint8x8x2_t ret;
      asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_s8_x2(a: *const i8) -> int8x8x2_t {
    
    todo!();
        /*
            int8x8x2_t ret;
      asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_u16_x2(a: *const u16) -> uint16x4x2_t {
    
    todo!();
        /*
            uint16x4x2_t ret;
      asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_s16_x2(a: *const i16) -> int16x4x2_t {
    
    todo!();
        /*
            int16x4x2_t ret;
      asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_u32_x2(a: *const u32) -> uint32x2x2_t {
    
    todo!();
        /*
            uint32x2x2_t ret;
      asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_s32_x2(a: *const i32) -> int32x2x2_t {
    
    todo!();
        /*
            int32x2x2_t ret;
      asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_u64_x2(a: *const u64) -> uint64x1x2_t {
    
    todo!();
        /*
            uint64x1x2_t ret;
      asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_s64_x2(a: *const i64) -> int64x1x2_t {
    
    todo!();
        /*
            int64x1x2_t ret;
      __builtin_aarch64_simd_oi __o;
      asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_f16_x2(a: *const float16_t) -> float16x4x2_t {
    
    todo!();
        /*
            float16x4x2_t ret;
      asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_f32_x2(a: *const f32) -> float32x2x2_t {
    
    todo!();
        /*
            float32x2x2_t ret;
      asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_f64_x2(a: *const float64_t) -> float64x1x2_t {
    
    todo!();
        /*
            float64x1x2_t ret;
      asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_p8_x2(a: *const poly8_t) -> poly8x8x2_t {
    
    todo!();
        /*
            poly8x8x2_t ret;
      asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_p16_x2(a: *const poly16_t) -> poly16x4x2_t {
    
    todo!();
        /*
            poly16x4x2_t ret;
      asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1_p64_x2(a: *const poly64_t) -> poly64x1x2_t {
    
    todo!();
        /*
            poly64x1x2_t ret;
      asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_u8_x2(a: *const u8) -> uint8x16x2_t {
    
    todo!();
        /*
            uint8x16x2_t ret;
      asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_s8_x2(a: *const i8) -> int8x16x2_t {
    
    todo!();
        /*
            int8x16x2_t ret;
      asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_u16_x2(a: *const u16) -> uint16x8x2_t {
    
    todo!();
        /*
            uint16x8x2_t ret;
      asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_s16_x2(a: *const i16) -> int16x8x2_t {
    
    todo!();
        /*
            int16x8x2_t ret;
      asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_u32_x2(a: *const u32) -> uint32x4x2_t {
    
    todo!();
        /*
            uint32x4x2_t ret;
      asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_s32_x2(a: *const i32) -> int32x4x2_t {
    
    todo!();
        /*
            int32x4x2_t ret;
      asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_u64_x2(a: *const u64) -> uint64x2x2_t {
    
    todo!();
        /*
            uint64x2x2_t ret;
      asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_s64_x2(a: *const i64) -> int64x2x2_t {
    
    todo!();
        /*
            int64x2x2_t ret;
      asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_f16_x2(a: *const float16_t) -> float16x8x2_t {
    
    todo!();
        /*
            float16x8x2_t ret;
      asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_f32_x2(a: *const f32) -> float32x4x2_t {
    
    todo!();
        /*
            float32x4x2_t ret;
      asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_f64_x2(a: *const float64_t) -> float64x2x2_t {
    
    todo!();
        /*
            float64x2x2_t ret;
      asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_p8_x2(a: *const poly8_t) -> poly8x16x2_t {
    
    todo!();
        /*
            poly8x16x2_t ret;
      asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_p16_x2(a: *const poly16_t) -> poly16x8x2_t {
    
    todo!();
        /*
            poly16x8x2_t ret;
      asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

#[inline(always)] pub fn vld1q_p64_x2(a: *const poly64_t) -> poly64x2x2_t {
    
    todo!();
        /*
            poly64x2x2_t ret;
      asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
      return ret;
        */
}

/* --------------------- vst1x2  --------------------- */
#[inline(always)] pub fn vst1_s64_x2(
        a:   *mut i64,
        val: int64x1x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_u64_x2(
        a:   *mut u64,
        val: uint64x1x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_f64_x2(
        a:   *mut float64_t,
        val: float64x1x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_s8_x2(
        a:   *mut i8,
        val: int8x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_p8_x2(
        a:   *mut poly8_t,
        val: poly8x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_s16_x2(
        a:   *mut i16,
        val: int16x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_p16_x2(
        a:   *mut poly16_t,
        val: poly16x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_s32_x2(
        a:   *mut i32,
        val: int32x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_u8_x2(
        a:   *mut u8,
        val: uint8x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_u16_x2(
        a:   *mut u16,
        val: uint16x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_u32_x2(
        a:   *mut u32,
        val: uint32x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_f16_x2(
        a:   *mut float16_t,
        val: float16x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_f32_x2(
        a:   *mut f32,
        val: float32x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1_p64_x2(
        a:   *mut poly64_t,
        val: poly64x1x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_s8_x2(
        a:   *mut i8,
        val: int8x16x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_p8_x2(
        a:   *mut poly8_t,
        val: poly8x16x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_s16_x2(
        a:   *mut i16,
        val: int16x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_p16_x2(
        a:   *mut poly16_t,
        val: poly16x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_s32_x2(
        a:   *mut i32,
        val: int32x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_s64_x2(
        a:   *mut i64,
        val: int64x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_u8_x2(
        a:   *mut u8,
        val: uint8x16x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_u16_x2(
        a:   *mut u16,
        val: uint16x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_u32_x2(
        a:   *mut u32,
        val: uint32x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_u64_x2(
        a:   *mut u64,
        val: uint64x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_f16_x2(
        a:   *mut float16_t,
        val: float16x8x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_f32_x2(
        a:   *mut f32,
        val: float32x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_f64_x2(
        a:   *mut float64_t,
        val: float64x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
        */
}

#[inline(always)] pub fn vst1q_p64_x2(
        a:   *mut poly64_t,
        val: poly64x2x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
        */
}
