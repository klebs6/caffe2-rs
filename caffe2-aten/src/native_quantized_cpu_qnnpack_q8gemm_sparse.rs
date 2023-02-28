// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gemm_sparse.h]

#[macro_export] macro_rules! declare_pytorch_q8gemm_sparse_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              const u8* a,                          
              usize a_stride,                           
              const u8* packed_w,                   
              const u32* w_row_ptr,                 
              const u32* w_block_ids_ptr,           
              u8* c,                                
              usize c_stride,                           
              usize output_channel_index,               
              const union pytorch_qnnp_conv_quantization_params* quantization_params);
        */
    }
}


declare_pytorch_q8gemm_sparse_ukernel_function!{pytorch_q8gemm_sparse_1x4_ukernel_4x8__neon}
declare_pytorch_q8gemm_sparse_ukernel_function!{pytorch_q8gemm_sparse_1x4_ukernel_8x8__neon}
declare_pytorch_q8gemm_sparse_ukernel_function!{pytorch_q8gemm_sparse_1x4_ukernel_4x8__aarch32_neon}
declare_pytorch_q8gemm_sparse_ukernel_function!{pytorch_q8gemm_sparse_1x4_ukernel_8x8__aarch64_neon}
declare_pytorch_q8gemm_sparse_ukernel_function!{pytorch_q8gemm_sparse_1x4_ukernel_4x4c2__sse2}

#[macro_export] macro_rules! declare_pytorch_q8gemm_dynamic_quantization_sparse_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              const u8* a,                          
              usize a_stride,                           
              const u8* packed_w,                   
              const u32* w_row_ptr,                 
              const u32* w_block_ids_ptr,           
              const float* b,                            
              float* c,                                  
              usize c_stride,                           
              usize output_channel_index,               
              const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);
        */
    }
}


declare_pytorch_q8gemm_dynamic_quantization_sparse_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch64_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__sse2}

#[macro_export] macro_rules! declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              usize mr,                                 
              usize nr,                                 
              const u8* a_packed,                   
              const u8* packed_w,                   
              const u32* w_row_ptr,                 
              const u32* w_block_ids_ptr,           
              const float* b,                            
              float* c,                                  
              usize c_stride,                           
              usize output_channel_index,               
              const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);
        */
    }
}

declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA__aarch32_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA__aarch64_neon}
declare_pytorch_q8gemm_dynamic_quantization_sparse_packeda_ukernel_function!{pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__sse2}

#[macro_export] macro_rules! declare_pytorch_q8gemm_parse_packa_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(            
              const usize mr,                           
              const usize K,                            
              const u8* a,                          
              const usize a_stride,                     
              u8* a_packed);
        */
    }
}

declare_pytorch_q8gemm_parse_packa_ukernel_function!{pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon}
declare_pytorch_q8gemm_parse_packa_ukernel_function!{pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon}
declare_pytorch_q8gemm_parse_packa_ukernel_function!{pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon}
declare_pytorch_q8gemm_parse_packa_ukernel_function!{pytorch_q8gemm_sparse_packA_ukernel_8x4__sse2}
