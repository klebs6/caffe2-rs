crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8dwconv.h]

#[macro_export] macro_rules! declare_pytorch_q8updwconv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                
              usize channels,                               
              usize output_width,                           
              const u8** input,                         
              const void* weights,                           
              u8* output,                               
              usize input_stride,                           
              usize output_increment,                       
              const union pytorch_qnnp_conv_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9__neon}
declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9_per_channel__neon}
declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9__aarch32_neon}
declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon}
declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9__sse2}
declare_pytorch_q8updwconv_ukernel_function!{pytorch_q8dwconv_ukernel_up8x9_per_channel__sse2}

#[macro_export] macro_rules! declare_pytorch_q8mpdwconv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                
              usize channels,                               
              usize output_width,                           
              const u8** input,                         
              const void* weights,                           
              i32* buffer,                               
              u8* output,                               
              usize input_stride,                           
              usize output_increment,                       
              const union pytorch_qnnp_conv_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8mpdwconv_ukernel_function!{pytorch_q8dwconv_ukernel_mp8x25__neon}
declare_pytorch_q8mpdwconv_ukernel_function!{pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon}
declare_pytorch_q8mpdwconv_ukernel_function!{pytorch_q8dwconv_ukernel_mp8x25__sse2}
declare_pytorch_q8mpdwconv_ukernel_function!{pytorch_q8dwconv_ukernel_mp8x25_per_channel__sse2}
