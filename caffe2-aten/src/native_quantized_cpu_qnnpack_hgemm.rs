crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/hgemm.h]

#[macro_export] macro_rules! declare_pytorch_hgemm_ukernel_function {
    ($fn_name:ident) => {
        /*
        
          void fn_name(                                 
              usize mr,                                
              usize nr,                                
              usize k,                                 
              const void* a,                            
              usize a_stride,                          
              const void* w,                            
              void* c,                                  
              usize c_stride,                          
              const struct pytorch_qnnp_fp16_clamping_params* clamping_params);
        */
    }
}

declare_pytorch_hgemm_ukernel_function!{pytorch_hgemm_ukernel_8x8__neonfp16arith}
declare_pytorch_hgemm_ukernel_function!{pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith}
