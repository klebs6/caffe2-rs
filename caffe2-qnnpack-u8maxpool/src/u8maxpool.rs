crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/u8maxpool.h]

#[macro_export] macro_rules! declare_pytorch_u8maxpool_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          PYTORCH_QNNP_INTERNAL void fn_name(               
              usize n,                                     
              usize ks,                                    
              usize kc,                                    
              const u8** x,                            
              u8* y,                                   
              usize x_increment,                           
              usize y_increment,                           
              const union pytorch_qnnp_u8_clamping_params* params);
        */
    }
}


declare_pytorch_u8maxpool_ukernel_function!{pytorch_u8maxpool_ukernel_16x9p8q__neon}
declare_pytorch_u8maxpool_ukernel_function!{pytorch_u8maxpool_ukernel_16x9p8q__sse2}
declare_pytorch_u8maxpool_ukernel_function!{pytorch_u8maxpool_ukernel_sub16__neon}
declare_pytorch_u8maxpool_ukernel_function!{pytorch_u8maxpool_ukernel_sub16__sse2}
