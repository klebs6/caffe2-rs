crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/q8avgpool.h]

#[macro_export] macro_rules! declare_pytorch_q8mpavgpool_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                       
              usize n,                                             
              usize ks,                                            
              usize kc,                                            
              const u8** x,                                    
              const u8* zero,                                  
              i32* buffer,                                      
              u8* y,                                           
              usize x_increment,                                   
              usize y_increment,                                   
              const union pytorch_qnnp_avgpool_quantization_params* 
                  quantization_params);
        */
    }
}

declare_pytorch_q8mpavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_mp8x9p8q__neon}
declare_pytorch_q8mpavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_mp8x9p8q__sse2}

#[macro_export] macro_rules! declare_pytorch_q8upavgpool_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(                       
              usize n,                                             
              usize ks,                                            
              usize kc,                                            
              const u8** x,                                    
              const u8* zero,                                  
              u8* y,                                           
              usize x_increment,                                   
              usize y_increment,                                   
              const union pytorch_qnnp_avgpool_quantization_params* 
                  quantization_params);
        */
    }
}


declare_pytorch_q8upavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_up8x9__neon}
declare_pytorch_q8upavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_up8xm__neon}
declare_pytorch_q8upavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_up8x9__sse2}
declare_pytorch_q8upavgpool_ukernel_function!{pytorch_q8avgpool_ukernel_up8xm__sse2}
