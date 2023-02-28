// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/runtime-neon.h]

#[inline]
pub fn sub_zero_point(
        va:  uint8x8_t,
        vzp: uint8x8_t) -> uint16x8_t {
    
    todo!();
        /*
            #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
      // Run-time quantization
      return vsubl_u8(va, vzp);
    #else
      // Design-time quantization
      return vmovl_u8(va);
    #endif
        */
}
