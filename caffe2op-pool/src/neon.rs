crate::ix!();

#[inline] pub fn is_neon4x4_p0s0_eligible(
    input_h:      i32,
    input_w:      i32,
    output_h:     i32,
    output_w:     i32,
    kh:           i32,
    kw:           i32,
    stride_h:     i32,
    stride_w:     i32,
    pad_t:        i32,
    pad_l:        i32,
    pad_b:        i32,
    pad_r:        i32,
    dilation_h:   i32,
    dilation_w:   i32,
    x:            *const f32,
    y:            *mut f32) -> bool {

    todo!();
    /*
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      // Use this kernel only if:
      //   1. Kernel size is 4x4
      //   2. Stride is 4x4
      //   3. Padding is 0
      //   4. Dilation is 1
      //   5. Output width and height are even divisors of input width
      //   6. Input width and height are divisible by 4 (should be implied by all of
      //      the above, but just check again)
      // Input and output pointers are aligned by float32x4_t
      const bool kernel_ok = (kh == 4) && (kw == 4);
      const bool stride_ok = (stride_h == 4) && (stride_w == 4);
      const bool pad_ok =
          (pad_t == 0) && (pad_l == 0) && (pad_b == 0) && (pad_r == 0);
      const bool dilation_ok = (dilation_h == 1) && (dilation_w == 1);
      const bool output_ok = (input_h % output_h == 0) && (input_w % output_w == 0);
      const bool input_ok = (input_w % 4 == 0) && (input_h % 4 == 0);
      const bool align_ok = isPointerAligned(X, sizeof(float32x4_t)) &&
          isPointerAligned(Y, sizeof(float32x4_t));
      return kernel_ok && stride_ok && pad_ok && dilation_ok && output_ok &&
          input_ok && align_ok;
    #else
      (void)input_h;
      (void)input_w;
      (void)output_h;
      (void)output_w;
      (void)kh;
      (void)kw;
      (void)stride_h;
      (void)stride_w;
      (void)pad_t;
      (void)pad_l;
      (void)pad_b;
      (void)pad_r;
      (void)dilation_h;
      (void)dilation_w;
      (void)X;
      (void)Y;
      return false;
    #endif
    */
}

#[inline] pub fn is_neon2x2p0s_0eligible(
    input_h:      i32,
    input_w:      i32,
    output_h:     i32,
    output_w:     i32,
    kh:           i32,
    kw:           i32,
    stride_h:     i32,
    stride_w:     i32,
    pad_t:        i32,
    pad_l:        i32,
    pad_b:        i32,
    pad_r:        i32,
    dilation_h:   i32,
    dilation_w:   i32,
    x:            *const f32,
    y:            *mut f32) -> bool 
{
    todo!();
    /*
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      // Use this kernel only if:
      //   1. Kernel size is 2x2
      //   2. Stride is 2x2
      //   3. Padding is 0
      //   4. Dilation is 1
      //   5. Output width and height are even divisors of input width
      //   6. Input width and height are divisible by 4 (should be implied b all of
      //      the above, but just check again)
      // Input and output pointers are aligned by float32x4_t
      const bool kernel_ok = (kh == 2) && (kw == 2);
      const bool stride_ok = (stride_h == 2) && (stride_w == 2);
      const bool pad_ok =
          (pad_t == 0) && (pad_l == 0) && (pad_b == 0) && (pad_r == 0);
      const bool dilation_ok = (dilation_h == 1) && (dilation_w == 1);
      const bool output_ok = (input_h % output_h == 0) && (input_w % output_w == 0);
      const bool input_ok = (input_w % 4 == 0) && (input_h % 4 == 0);
      const bool align_ok = isPointerAligned(X, sizeof(float32x4_t)) &&
          isPointerAligned(Y, sizeof(float32x4_t));
      return kernel_ok && stride_ok && pad_ok && dilation_ok && output_ok &&
          input_ok && align_ok;
    #else
      (void)input_h;
      (void)input_w;
      (void)output_h;
      (void)output_w;
      (void)kh;
      (void)kw;
      (void)stride_h;
      (void)stride_w;
      (void)pad_t;
      (void)pad_l;
      (void)pad_b;
      (void)pad_r;
      (void)dilation_h;
      (void)dilation_w;
      (void)X;
      (void)Y;
      return false;
    #endif
    */
}
