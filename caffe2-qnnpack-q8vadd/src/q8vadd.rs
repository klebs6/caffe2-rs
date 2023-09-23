// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8vadd.h]

#[macro_export] macro_rules! declare_pytorch_q8vadd_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize n,                                  
              const u8* a,                          
              const u8* b,                          
              u8* y,                                
              const union pytorch_qnnp_add_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8vadd_ukernel_function!{pytorch_q8vadd_ukernel__neon}
declare_pytorch_q8vadd_ukernel_function!{pytorch_q8vadd_ukernel__sse2}
