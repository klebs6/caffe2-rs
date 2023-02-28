crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8gavgpool.h]

#[macro_export] macro_rules! declare_pytorch_q8mpgavgpool_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                       
              usize m,                                             
              usize n,                                             
              const u8* x,                                     
              usize x_stride,                                      
              const u8* zero,                                  
              i32* buffer,                                      
              u8* y,                                           
              const union pytorch_qnnp_avgpool_quantization_params* 
                  quantization_params);
        */
    }
}

declare_pytorch_q8mpgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_mp8x7p7q__neon}
declare_pytorch_q8mpgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_mp8x7p7q__sse2}

#[macro_export] macro_rules! declare_pytorch_q8upgavgpool_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                       
              usize m,                                             
              usize n,                                             
              const u8* x,                                     
              usize x_stride,                                      
              const u8* zero,                                  
              u8* y,                                           
              const union pytorch_qnnp_avgpool_quantization_params* 
                  quantization_params);
        */
    }
}


declare_pytorch_q8upgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_up8x7__neon}
declare_pytorch_q8upgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_up8xm__neon}
declare_pytorch_q8upgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_up8x7__sse2}
declare_pytorch_q8upgavgpool_ukernel_function!{pytorch_q8gavgpool_ukernel_up8xm__sse2}
