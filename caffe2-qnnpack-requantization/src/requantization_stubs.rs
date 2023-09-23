// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization-stubs.h]

pub type PytorchRequantizationFunction = fn(
        n:          usize,
        input:      *const i32,
        scale:      f32,
        zero_point: u8,
        qmin:       u8,
        qmax:       u8,
        output:     *mut u8
) -> void;

#[macro_export] macro_rules! declare_pytorch_requantization_function {
    ($fn_name:ident) => {
        /*
        
          void fn_name(                                  
              usize n,                                  
              const i32* input,                      
              float scale,                               
              u8 zero_point,                        
              u8 qmin,                              
              u8 qmax,                              
              u8* output);
        */
    }
}

declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__scalar_unsigned32}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__scalar_unsigned64}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__scalar_signed64}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__sse2}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__ssse3}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__sse4}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__neon}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_precise__psimd}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_fp32__scalar_lrintf}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_fp32__scalar_magic}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_fp32__sse2}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_fp32__neon}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_fp32__psimd}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__scalar}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__sse2}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__ssse3}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__sse4}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__neon}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_q31__psimd}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_gemmlowp__scalar}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_gemmlowp__sse2}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_gemmlowp__ssse3}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_gemmlowp__sse4}
declare_pytorch_requantization_function!{pytorch_qnnp_requantize_gemmlowp__neon}
