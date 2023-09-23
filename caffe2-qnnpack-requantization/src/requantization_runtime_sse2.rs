// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/runtime-sse2.h]

#[inline]
pub fn sub_zero_point(
        va:  __m128i,
        vzp: __m128i) -> __m128i {
    
    todo!();
        /*
            #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      // Run-time quantization
      return _mm_sub_epi16(va, vzp);
    #else
      // Design-time quantization (no-op)
      return va;
    #endif
        */
}
