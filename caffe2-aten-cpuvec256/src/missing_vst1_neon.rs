//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/missing_vst1_neon.h]
/* Workaround for missing vst1q_f32_x2 in gcc-8.  */

//#[__extension__] 
#[inline(always)]
#[inline(always)] pub fn vst1q_f32_x2(
        a:   *mut f32,
        val: float32x4x2_t)  {
    
    todo!();
        /*
            asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
        */
}
