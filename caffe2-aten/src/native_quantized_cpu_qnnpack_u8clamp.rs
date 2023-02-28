crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/u8clamp.h]

#[macro_export] macro_rules! declare_pytorch_u8clamp_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(             
              usize n,                                   
              const u8* x,                           
              u8* y,                                 
              const union pytorch_qnnp_u8_clamping_params* params);
        */
    }
}

declare_pytorch_u8clamp_ukernel_function!{pytorch_u8clamp_ukernel__neon}
declare_pytorch_u8clamp_ukernel_function!{pytorch_u8clamp_ukernel__sse2}
