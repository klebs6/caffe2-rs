// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm.h]

#[macro_export] macro_rules! declare_pytorch_q8gemm_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              usize k,                                  
              const u8* a,                          
              usize a_stride,                           
              const void* w,                             
              u8* c,                                
              usize c_stride,                           
              usize output_channel_index,               
              const union pytorch_qnnp_conv_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_3x3c8__neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_2x4c8__neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_4x8__neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_6x4__neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_8x8__neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_4x8__aarch32_neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_8x8__aarch64_neon}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_2x4c8__sse2}
declare_pytorch_q8gemm_ukernel_function!{pytorch_q8gemm_ukernel_4x4c2__sse2}

#[macro_export] macro_rules! declare_pytorch_q8gemm_dynamic_quantization_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              usize k,                                  
              const u8* a,                          
              usize a_stride,                           
              const void* w,                             
              const float* b,                            
              float* c,                                  
              usize c_stride,                           
              usize output_channel_index,               
              const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);
        */
    }
}


declare_pytorch_q8gemm_dynamic_quantization_ukernel_function!{pytorch_q8gemm_dq_ukernel_4x8__neon}
declare_pytorch_q8gemm_dynamic_quantization_ukernel_function!{pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon}
declare_pytorch_q8gemm_dynamic_quantization_ukernel_function!{pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon}
declare_pytorch_q8gemm_dynamic_quantization_ukernel_function!{pytorch_q8gemm_dq_ukernel_4x4c2__sse2}

#[macro_export] macro_rules! declare_pytorch_q8gemm_xzp_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                     
              usize mr,                                          
              usize nr,                                          
              usize k,                                           
              const u8* a,                                   
              usize a_stride,                                    
              const i32* a_sum,                               
              const void* w,                                      
              u8* c,                                         
              usize c_stride,                                    
              const union pytorch_qnnp_q31_requantization_params* 
                  requantization_params);
        */
    }
}

declare_pytorch_q8gemm_xzp_ukernel_function!{pytorch_q8gemm_xzp_ukernel_4x8c2__neon}
declare_pytorch_q8gemm_xzp_ukernel_function!{pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon}

pub fn pytorch_q8sumrows_ukernel_4x_neon(
        a:          *const u8,
        m:          usize,
        k:          usize,
        stride:     usize,
        multiplier: i32,
        row_sum:    *mut i32)  {
    
    todo!();
        /*
        
        */
}
