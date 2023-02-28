crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8conv.h]

#[macro_export] macro_rules! declare_pytorch_q8conv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              usize kc,                                 
              usize ks,                                 
              const u8** a,                         
              const void* w,                             
              u8* c,                                
              usize c_stride,                           
              usize output_channel_index,               
              const union pytorch_qnnp_conv_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8conv_ukernel_function!{pytorch_q8conv_ukernel_4x8__neon}
declare_pytorch_q8conv_ukernel_function!{pytorch_q8conv_ukernel_4x8__aarch32_neon}
declare_pytorch_q8conv_ukernel_function!{pytorch_q8conv_ukernel_8x8__aarch64_neon}
declare_pytorch_q8conv_ukernel_function!{pytorch_q8conv_ukernel_8x8__neon}
declare_pytorch_q8conv_ukernel_function!{pytorch_q8conv_ukernel_4x4c2__sse2}
