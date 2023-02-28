crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/sdwconv.h]

#[macro_export] macro_rules! declare_pytorch_supdwconv_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(               
              usize channels,                              
              usize output_width,                          
              const float** input,                          
              const float* weights,                         
              float* output,                                
              usize input_stride,                          
              usize output_increment,                      
              const struct pytorch_qnnp_fp32_clamping_params* clamping_params);
        */
    }
}

declare_pytorch_supdwconv_ukernel_function!{pytorch_sdwconv_ukernel_up4x9__psimd}

#[macro_export] macro_rules! declare_pytorch_smpdwconv_ukernel_function {
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
              const struct pytorch_qnnp_fp32_clamping_params* clamping_params);
        */
    }
}
